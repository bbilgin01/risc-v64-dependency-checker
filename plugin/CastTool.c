/*
  CastTool.c - LEJ Chain Tracker for MAMBO (RISC-V64)
  - Single-thread only (no TLS, no locks, no pre/post thread)
  - Simple heap allocation only for UNIQUE chain entries (malloc)
  - Runtime analysis called via emit_safe_fcall (argument passed in x10/a0)

  Tracks:
    total dynamic instructions
    total long (load+store) executed
    total expensive executed (mul/div/rem + W variants)
    unique LEJ chains with occurrence count
    first_seen dynamic id (join seq)

  Output: chain.txt
*/

#ifdef PLUGINS_NEW

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "../plugins.h"

// -----------------------------
// Instruction meta (scan-time allocated; read at runtime)
// -----------------------------

typedef struct {
  uintptr_t pc;       // guest/original PC
  uint16_t  inst;     // decoded enum (MAMBO)
  uint32_t  raw32;    // raw encoding (best-effort; only if len==4)
  uint8_t   rd, rs1, rs2;
  uint8_t   flags;
} ins_meta_t;

enum {
  F_IS_LOAD      = 1u << 0,
  F_IS_STORE     = 1u << 1,
  F_IS_EXPENSIVE = 1u << 2,
  F_WRITES_RD    = 1u << 3,
  F_HAS_RS1      = 1u << 4,
  F_HAS_RS2      = 1u << 5,
  F_IS_JOIN_CAND = 1u << 6
};

// -----------------------------
// Producer tracking
// -----------------------------

typedef enum {
  PROD_NONE = 0,
  PROD_LONG_LOAD,
  PROD_EXPENSIVE,
  PROD_OTHER
} prod_kind_t;

typedef struct {
  prod_kind_t kind;
  uint64_t seq;
  const ins_meta_t *meta;
  uint8_t rd, rs1, rs2;
} producer_t;

// -----------------------------
// Chain hash map (PC-based key)
// -----------------------------

#ifndef LEJ_BUCKETS
#define LEJ_BUCKETS 4096u  // power of 2
#endif

typedef struct {
  uintptr_t pcL, pcE, pcJ;
  uint16_t  instL, instE, instJ;
  uint8_t   rdL,  rs1L, rs2L;
  uint8_t   rdE,  rs1E, rs2E;
  uint8_t   rdJ,  rs1J, rs2J;
} chain_key_t;

typedef struct chain_entry {
  chain_key_t key;
  uint64_t count;
  uint64_t first_dyn_id;
  struct chain_entry *next;
} chain_entry_t;

typedef struct {
  chain_entry_t *buckets[LEJ_BUCKETS];
} chain_map_t;

static inline uint64_t mix64(uint64_t x) {
  // splitmix64 finalizer
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static inline uint32_t chain_hash(const chain_key_t *k) {
  uint64_t h = 0;
  h ^= mix64((uint64_t)k->pcL);
  h ^= mix64((uint64_t)k->pcE + 0x1111);
  h ^= mix64((uint64_t)k->pcJ + 0x2222);
  h ^= mix64(((uint64_t)k->instL << 32) | (uint64_t)k->instE);
  h ^= mix64(((uint64_t)k->instJ << 32) |
             ((uint64_t)k->rdJ << 16) |
             ((uint64_t)k->rs1J << 8) |
             (uint64_t)k->rs2J);
  return (uint32_t)(h & (LEJ_BUCKETS - 1));
}

static inline bool chain_eq(const chain_key_t *a, const chain_key_t *b) {
  return memcmp(a, b, sizeof(chain_key_t)) == 0;
}

static void chain_map_init(chain_map_t *m) {
  memset(m, 0, sizeof(*m));
}

static void chain_map_inc(chain_map_t *m, const chain_key_t *key, uint64_t dyn_id) {
  const uint32_t b = chain_hash(key);

  for (chain_entry_t *e = m->buckets[b]; e; e = e->next) {
    if (chain_eq(&e->key, key)) {
      e->count++;
      return;
    }
  }

  chain_entry_t *ne = (chain_entry_t *)malloc(sizeof(chain_entry_t));
  if (!ne) return;

  ne->key = *key;
  ne->count = 1;
  ne->first_dyn_id = dyn_id;
  ne->next = m->buckets[b];
  m->buckets[b] = ne;
}

static uint64_t chain_map_total(const chain_map_t *m) {
  uint64_t t = 0;
  for (uint32_t i = 0; i < LEJ_BUCKETS; i++)
    for (const chain_entry_t *e = m->buckets[i]; e; e = e->next)
      t += e->count;
  return t;
}

static uint64_t chain_map_unique(const chain_map_t *m) {
  uint64_t u = 0;
  for (uint32_t i = 0; i < LEJ_BUCKETS; i++)
    for (const chain_entry_t *e = m->buckets[i]; e; e = e->next)
      u++;
  return u;
}

// -----------------------------
// Minimal decode helpers (only for 32-bit)
// -----------------------------

static inline uint32_t load_u32(const void *p) {
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

static inline uint32_t bits(uint32_t x, int hi, int lo) {
  return (x >> lo) & ((1u << (hi - lo + 1)) - 1u);
}

static void decode_rv_regs_and_kind(uint32_t raw,
                                   uint8_t *rd, uint8_t *rs1, uint8_t *rs2,
                                   uint8_t *flags /* in/out */) {
  const uint32_t opc = bits(raw, 6, 0);
  *rd = *rs1 = *rs2 = 0;

  switch (opc) {
    case 0x03: // LOAD (I-type)
      *rd  = (uint8_t)bits(raw, 11, 7);
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *flags |= F_WRITES_RD | F_HAS_RS1;
      break;

    case 0x23: // STORE (S-type)
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *rs2 = (uint8_t)bits(raw, 24, 20);
      *flags |= F_HAS_RS1 | F_HAS_RS2;
      break;

    case 0x33: // OP (R-type)
    case 0x3B: // OP-32 (R-type)
      *rd  = (uint8_t)bits(raw, 11, 7);
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *rs2 = (uint8_t)bits(raw, 24, 20);
      *flags |= F_WRITES_RD | F_HAS_RS1 | F_HAS_RS2 | F_IS_JOIN_CAND;
      break;

    case 0x13: // OP-IMM
    case 0x1B: // OP-IMM-32
      *rd  = (uint8_t)bits(raw, 11, 7);
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *flags |= F_WRITES_RD | F_HAS_RS1;
      break;

    default:
      break;
  }
}

static inline bool is_expensive_enum(uint16_t inst) {
  switch (inst) {
    case RISCV_MUL:
    case RISCV_MULH:
    case RISCV_MULHSU:
    case RISCV_MULHU:
    case RISCV_DIV:
    case RISCV_DIVU:
    case RISCV_REM:
    case RISCV_REMU:
    case RISCV_MULW:
    case RISCV_DIVW:
    case RISCV_DIVUW:
    case RISCV_REMW:
    case RISCV_REMUW:
      return true;
    default:
      return false;
  }
}

// -----------------------------
// Printing helpers (single-thread => simple static buffers)
// -----------------------------

static inline const char *reg_name(uint8_t r) {
  static char b[32][8];
  if (r < 32) {
    snprintf(b[r], sizeof(b[r]), "r%u", (unsigned)r);
    return b[r];
  }
  return "r?";
}

static inline const char *inst_name(uint16_t inst) {
  switch (inst) {
    case RISCV_LD: return "ld";
    case RISCV_LW: return "lw";
    case RISCV_SD: return "sd";
    case RISCV_SW: return "sw";
    case RISCV_ADD: return "add";
    case RISCV_SUB: return "sub";
    case RISCV_AND: return "and";
    case RISCV_OR:  return "or";
    case RISCV_XOR: return "xor";
    case RISCV_MUL: return "mul";
    case RISCV_DIV: return "div";
    case RISCV_DIVU: return "divu";
    case RISCV_REM: return "rem";
    case RISCV_REMU: return "remu";
    case RISCV_MULW: return "mulw";
    case RISCV_DIVW: return "divw";
    case RISCV_DIVUW: return "divuw";
    case RISCV_REMW: return "remw";
    case RISCV_REMUW: return "remuw";
    default: return "inst";
  }
}

static void fmt_ins(char *out, size_t n,
                    uint16_t inst, uint8_t rd, uint8_t rs1, uint8_t rs2, uint8_t flags) {
  const char *mn = inst_name(inst);

  if (flags & F_IS_LOAD) {
    snprintf(out, n, "%s %s", mn, reg_name(rd));
    return;
  }
  if (flags & F_IS_STORE) {
    snprintf(out, n, "%s %s", mn, reg_name(rs2));
    return;
  }
  if ((flags & F_HAS_RS1) && (flags & F_HAS_RS2)) {
    snprintf(out, n, "%s %s %s %s", mn, reg_name(rd), reg_name(rs1), reg_name(rs2));
    return;
  }
  if (flags & F_HAS_RS1) {
    snprintf(out, n, "%s %s %s", mn, reg_name(rd), reg_name(rs1));
    return;
  }
  snprintf(out, n, "%s", mn);
}
static uint64_t   g_dyn_seq     = 0;
static uint64_t   g_total_insts = 0;
static uint64_t   g_long_insts  = 0;
static uint64_t   g_exp_insts   = 0;

static producer_t g_last_prod[32];
static chain_map_t g_chains;

// -----------------------------
// Runtime analysis (called from emitted code)
// -----------------------------

__attribute__((noinline)) static void lej_analyze(const ins_meta_t *m) {
  if (!m) return;

  const uint64_t cur_seq = ++g_dyn_seq;
  g_total_insts++;

  const bool is_load  = (m->flags & F_IS_LOAD) != 0;
  const bool is_store = (m->flags & F_IS_STORE) != 0;
  const bool is_long  = is_load || is_store;
  const bool is_exp   = (m->flags & F_IS_EXPENSIVE) != 0;

  if (is_long) g_long_insts++;
  if (is_exp)  g_exp_insts++;

  // JOIN detection: require rs1+rs2
  if ((m->flags & F_IS_JOIN_CAND) && (m->flags & F_HAS_RS1) && (m->flags & F_HAS_RS2)) {
    const uint8_t rs1 = m->rs1;
    const uint8_t rs2 = m->rs2;

    if (rs1 < 32 && rs2 < 32) {
      const producer_t p1 = g_last_prod[rs1];
      const producer_t p2 = g_last_prod[rs2];

      const bool p1L = (p1.kind == PROD_LONG_LOAD);
      const bool p2L = (p2.kind == PROD_LONG_LOAD);
      const bool p1E = (p1.kind == PROD_EXPENSIVE);
      const bool p2E = (p2.kind == PROD_EXPENSIVE);

      if ((p1L && p2E) || (p2L && p1E)) {
        const producer_t PL = p1L ? p1 : p2;
        const producer_t PE = p1E ? p1 : p2;

        if (PL.meta && PE.meta && PL.seq < cur_seq && PE.seq < cur_seq) {
          // Independence: E must not read L.rd; L must not read E.rd
          const bool indep =
              (PE.rs1 != PL.rd && PE.rs2 != PL.rd) &&
              (PL.rs1 != PE.rd && PL.rs2 != PE.rd);

          if (indep) {
            chain_key_t k;
            k.pcL = PL.meta->pc;
            k.pcE = PE.meta->pc;
            k.pcJ = m->pc;

            k.instL = PL.meta->inst;
            k.instE = PE.meta->inst;
            k.instJ = m->inst;

            k.rdL  = PL.meta->rd;  k.rs1L = PL.meta->rs1;  k.rs2L = PL.meta->rs2;
            k.rdE  = PE.meta->rd;  k.rs1E = PE.meta->rs1;  k.rs2E = PE.meta->rs2;
            k.rdJ  = m->rd;        k.rs1J = m->rs1;        k.rs2J = m->rs2;

            chain_map_inc(&g_chains, &k, cur_seq);
          }
        }
      }
    }
  }

  // Producer update if writes rd
  if ((m->flags & F_WRITES_RD) && m->rd != 0 && m->rd < 32) {
    producer_t *pd = &g_last_prod[m->rd];

    prod_kind_t kind = PROD_OTHER;
    if (is_load)      kind = PROD_LONG_LOAD;     // only LOAD produces L
    else if (is_exp)  kind = PROD_EXPENSIVE;     // expensive producer

    pd->kind = kind;
    pd->seq  = cur_seq;
    pd->meta = m;
    pd->rd   = m->rd;
    pd->rs1  = (m->flags & F_HAS_RS1) ? m->rs1 : 0xFF;
    pd->rs2  = (m->flags & F_HAS_RS2) ? m->rs2 : 0xFF;
  }
}

// -----------------------------
// Plugin callbacks (single-thread)
// -----------------------------

static int CastTool_pre_inst(mambo_context *ctx);
static int CastTool_exit(mambo_context *ctx);

__attribute__((constructor)) void CastTool_init_plugin(void) {
  mambo_context *ctx = mambo_register_plugin();
  assert(ctx != NULL);

  chain_map_init(&g_chains);
  memset(g_last_prod, 0, sizeof(g_last_prod));
  g_dyn_seq = g_total_insts = g_long_insts = g_exp_insts = 0;

  mambo_register_pre_inst_cb(ctx, &CastTool_pre_inst);
  mambo_register_exit_cb(ctx, &CastTool_exit);
}

static int CastTool_pre_inst(mambo_context *ctx) {
  // Allocate per-instruction meta in MAMBO memory (lifetime = translated code)
  ins_meta_t *m = (ins_meta_t *)mambo_alloc(ctx, sizeof(ins_meta_t));
  assert(m != NULL);
  memset(m, 0, sizeof(*m));

  m->pc   = (uintptr_t)mambo_get_source_addr(ctx);
  m->inst = (uint16_t)mambo_get_inst(ctx);

  uint8_t flags = 0;
  if (mambo_is_load(ctx))  flags |= F_IS_LOAD;
  if (mambo_is_store(ctx)) flags |= F_IS_STORE;
  if (is_expensive_enum(m->inst)) flags |= F_IS_EXPENSIVE;

  // Only decode operands for 32-bit instructions (keeps things simple & safe)
  const int ilen = mambo_get_inst_len(ctx);
  if (ilen == 4) {
    m->raw32 = load_u32(mambo_get_source_addr(ctx));
    decode_rv_regs_and_kind(m->raw32, &m->rd, &m->rs1, &m->rs2, &flags);
  } else {
    // compressed / unusual: skip operand decode (won't form joins)
    m->raw32 = 0;
    m->rd = m->rs1 = m->rs2 = 0;
  }

  m->flags = flags;

  // IMPORTANT: avoid emit_safe_fcall_static_args on RISC-V (register bug)
  // Pass arg in x10 (a0), then call
  emit_set_reg(ctx, x10, (uintptr_t)m);
  emit_safe_fcall(ctx, (void *)&lej_analyze, 1);

  return 0;
}

static void dump_chains(FILE *fp) {
  fprintf(fp, "\n==== UNIQUE LEJ CHAINS ====\n");

  for (uint32_t i = 0; i < LEJ_BUCKETS; i++) {
    for (const chain_entry_t *e = g_chains.buckets[i]; e; e = e->next) {
      char sL[96], sE[96], sJ[96];

      // minimal flags for formatting
      const uint8_t fL = F_IS_LOAD | F_WRITES_RD | F_HAS_RS1;
      const uint8_t fE = F_WRITES_RD | F_HAS_RS1 | F_HAS_RS2;
      const uint8_t fJ = F_WRITES_RD | F_HAS_RS1 | F_HAS_RS2;

      fmt_ins(sE, sizeof(sE), e->key.instE, e->key.rdE, e->key.rs1E, e->key.rs2E, fE);
      fmt_ins(sL, sizeof(sL), e->key.instL, e->key.rdL, e->key.rs1L, e->key.rs2L, fL);
      fmt_ins(sJ, sizeof(sJ), e->key.instJ, e->key.rdJ, e->key.rs1J, e->key.rs2J, fJ);

      fprintf(fp, "dyn_id(first_seen): %" PRIu64 " | count: %" PRIu64 "\n",
              e->first_dyn_id, e->count);

      fprintf(fp, "consumer: %s (pc=0x%" PRIxPTR ")\n", sJ, e->key.pcJ);
      fprintf(fp, "producer: %s (r%u) (pc=0x%" PRIxPTR ")\n", sE, (unsigned)e->key.rdE, e->key.pcE);
      fprintf(fp, "producer: %s (r%u) (pc=0x%" PRIxPTR ")\n", sL, (unsigned)e->key.rdL, e->key.pcL);

      fprintf(fp, "chain:\n%s\n%s\n%s\n\n", sE, sL, sJ);
    }
  }

  fprintf(fp, "Unique chains: %" PRIu64 "\n", chain_map_unique(&g_chains));
  fprintf(fp, "Total LEJ detections (sum of counts): %" PRIu64 "\n", chain_map_total(&g_chains));
}

static int CastTool_exit(mambo_context *ctx) {
  (void)ctx;

  FILE *fp = fopen("chain.txt", "w");
  if (!fp) fp = stderr;

  const uint64_t total_det = chain_map_total(&g_chains);
  const uint64_t unique    = chain_map_unique(&g_chains);

  fprintf(fp, "==== GLOBAL STATS (DYNAMIC) ====\n");
  fprintf(fp, "total_instructions_executed: %" PRIu64 "\n", g_total_insts);
  fprintf(fp, "total_long_executed (load+store): %" PRIu64 "\n", g_long_insts);
  fprintf(fp, "total_expensive_executed (mul/div/rem): %" PRIu64 "\n", g_exp_insts);
  fprintf(fp, "total_lej_detections: %" PRIu64 "\n", total_det);
  fprintf(fp, "unique_lej_chains: %" PRIu64 "\n", unique);

  dump_chains(fp);

  if (fp != stderr) fclose(fp);
  return 0;
}

#endif // PLUGINS_NEW
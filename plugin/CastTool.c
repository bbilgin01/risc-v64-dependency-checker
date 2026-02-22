/*
  LEJ Chain Tracker (RISC-V64) for MAMBO

  Runtime (dynamic) tracking:
    1) Detect LEJ chains during execution:
        L = load (produces a reg)
        E = expensive (mul/div/rem family, produces a reg)
        J = join instruction that reads BOTH produced regs (rs1/rs2)
       L and E must be independent; order doesn't matter; J must come after both.

    2) Maintain unique chain statistics:
       - Key uses PCs of L/E/J => different code locations become different chains.
       - For each chain: occurrence count + first_seen_dynamic_id (join's seq).

    3) Global stats:
       - total dynamic instructions executed
       - total long (load+store) executed
       - total expensive executed
       - total LEJ detections (sum of chain counts)
       - unique chains

  Output file:
    chain.txt

  IMPORTANT:
    - pre_inst callback only EMITS a call into translated code.
    - Real analysis happens at runtime in lej_analyze().
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


typedef struct {
  uintptr_t pc;       // guest/original PC of this instruction
  uint16_t  inst;     // ctx->code.inst (MAMBO decoded enum)
  uint32_t  raw32;    // raw encoding (best-effort, used to decode regs)
  uint8_t   rd;
  uint8_t   rs1;
  uint8_t   rs2;
  uint8_t   flags;    // bitfield
} ins_meta_t;

// flags
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
// Producer tracking (per thread)
// -----------------------------

typedef enum {
  PROD_NONE = 0,
  PROD_LONG_LOAD,
  PROD_EXPENSIVE,
  PROD_OTHER
} prod_kind_t;

typedef struct {
  prod_kind_t kind;
  uint64_t seq;              // dynamic sequence number (per-thread)
  const ins_meta_t *meta;    // pointer to producer meta (stable)
  uint8_t rd;
  uint8_t rs1;
  uint8_t rs2;
} producer_t;

// -----------------------------
// Chain key + hash map (PC-based)
// -----------------------------

#ifndef LEJ_BUCKETS
#define LEJ_BUCKETS 4096  // power of 2
#endif

typedef struct {
  uintptr_t pcL, pcE, pcJ;   // PC-based uniqueness
  uint16_t  instL, instE, instJ;
  uint8_t   rdL,  rs1L, rs2L;
  uint8_t   rdE,  rs1E, rs2E;
  uint8_t   rdJ,  rs1J, rs2J;
} chain_key_t;

typedef struct chain_entry {
  chain_key_t key;
  uint64_t count;
  uint64_t first_dyn_id;   // join dyn-id when first seen
  struct chain_entry *next;
} chain_entry_t;

typedef struct {
  chain_entry_t *buckets[LEJ_BUCKETS];
} chain_map_t;

static inline uint64_t mix64(uint64_t x) {
  // splitmix64
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
  h ^= mix64(((uint64_t)k->instJ << 32) | ((uint64_t)k->rdJ << 8) | (uint64_t)k->rs1J);

  h ^= mix64(((uint64_t)k->rdL  << 16) | ((uint64_t)k->rs1L << 8) | (uint64_t)k->rs2L);
  h ^= mix64(((uint64_t)k->rdE  << 16) | ((uint64_t)k->rs1E << 8) | (uint64_t)k->rs2E);

  return (uint32_t)(h & (LEJ_BUCKETS - 1));
}

static inline bool chain_eq(const chain_key_t *a, const chain_key_t *b) {
  return memcmp(a, b, sizeof(chain_key_t)) == 0;
}

static void chain_map_init(chain_map_t *m) {
  memset(m, 0, sizeof(*m));
}

static void chain_map_inc(chain_map_t *m, const chain_key_t *key, uint64_t dyn_id) {
  uint32_t b = chain_hash(key);
  chain_entry_t *e = m->buckets[b];
  while (e) {
    if (chain_eq(&e->key, key)) {
      e->count++;
      return;
    }
    e = e->next;
  }

  chain_entry_t *ne = (chain_entry_t *)malloc(sizeof(chain_entry_t));
  assert(ne);
  ne->key = *key;
  ne->count = 1;
  ne->first_dyn_id = dyn_id;
  ne->next = m->buckets[b];
  m->buckets[b] = ne;
}

static void chain_map_merge(chain_map_t *dst, const chain_map_t *src) {
  for (uint32_t i = 0; i < LEJ_BUCKETS; i++) {
    for (const chain_entry_t *e = src->buckets[i]; e; e = e->next) {
      // insert e->count times (simple); for efficiency you can implement inc_by
      for (uint64_t n = 0; n < e->count; n++) {
        chain_map_inc(dst, &e->key, e->first_dyn_id);
      }
    }
  }
}

static uint64_t chain_map_total_detections(const chain_map_t *m) {
  uint64_t t = 0;
  for (uint32_t i = 0; i < LEJ_BUCKETS; i++) {
    for (const chain_entry_t *e = m->buckets[i]; e; e = e->next) {
      t += e->count;
    }
  }
  return t;
}

static uint64_t chain_map_unique(const chain_map_t *m) {
  uint64_t u = 0;
  for (uint32_t i = 0; i < LEJ_BUCKETS; i++) {
    for (const chain_entry_t *e = m->buckets[i]; e; e = e->next) {
      u++;
    }
  }
  return u;
}

// -----------------------------
// Per-thread + Global stats
// -----------------------------

typedef struct {
  uint64_t dyn_seq;

  uint64_t total_insts;
  uint64_t long_insts;       // load+store
  uint64_t expensive_insts;  // mul/div/rem family

  producer_t last_prod[32];
  chain_map_t chains;
} thread_state_t;

static __thread thread_state_t *tls = NULL;

static struct {
  uint64_t total_insts;
  uint64_t long_insts;
  uint64_t expensive_insts;

  chain_map_t chains;
  volatile int lock;
} g;

static inline void lock_acquire(volatile int *l) {
  while (__sync_lock_test_and_set(l, 1)) { }
}
static inline void lock_release(volatile int *l) {
  __sync_lock_release(l);
}

// -----------------------------
// Minimal decode helpers (RISC-V 32-bit formats)
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
    case 0x03: { // LOAD (I-type)
      *rd  = (uint8_t)bits(raw, 11, 7);
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *flags |= F_WRITES_RD | F_HAS_RS1;
      break;
    }
    case 0x23: { // STORE (S-type)
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *rs2 = (uint8_t)bits(raw, 24, 20);
      *flags |= F_HAS_RS1 | F_HAS_RS2;
      break;
    }
    case 0x33:   // OP (R-type)
    case 0x3B: { // OP-32 (R-type)
      *rd  = (uint8_t)bits(raw, 11, 7);
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *rs2 = (uint8_t)bits(raw, 24, 20);
      *flags |= F_WRITES_RD | F_HAS_RS1 | F_HAS_RS2 | F_IS_JOIN_CAND;
      break;
    }
    case 0x13:   // OP-IMM
    case 0x1B: { // OP-IMM-32
      *rd  = (uint8_t)bits(raw, 11, 7);
      *rs1 = (uint8_t)bits(raw, 19, 15);
      *flags |= F_WRITES_RD | F_HAS_RS1;
      break;
    }
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
// Pretty printing helpers
// -----------------------------

static inline const char* reg_name(uint8_t r) {
  static __thread char buf[32][8];
  snprintf(buf[r], sizeof(buf[r]), "r%u", (unsigned)r);
  return buf[r];
}

static inline const char* inst_name(uint16_t inst) {
  // Minimal set; extend as needed
  switch (inst) {
    case RISCV_LD: return "ld";
    case RISCV_SD: return "sd";
    case RISCV_LW: return "lw";
    case RISCV_SW: return "sw";
    case RISCV_ADD: return "add";
    case RISCV_SUB: return "sub";
    case RISCV_AND: return "and";
    case RISCV_OR:  return "or";
    case RISCV_XOR: return "xor";
    case RISCV_SLT: return "slt";
    case RISCV_SLTU:return "sltu";
    case RISCV_MUL: return "mul";
    case RISCV_DIV: return "div";
    case RISCV_DIVU:return "divu";
    case RISCV_REM: return "rem";
    case RISCV_REMU:return "remu";
    case RISCV_MULW:return "mulw";
    case RISCV_DIVW:return "divw";
    case RISCV_DIVUW:return "divuw";
    case RISCV_REMW:return "remw";
    case RISCV_REMUW:return "remuw";
    default: return "inst";
  }
}

static void fmt_ins(char *out, size_t n, const ins_meta_t *m) {
  const char *mn = inst_name(m->inst);

  if (m->flags & F_IS_LOAD) {
    // User asked: "ld r1" style (no imm/base)
    snprintf(out, n, "%s %s", mn, reg_name(m->rd));
    return;
  }
  if (m->flags & F_IS_STORE) {
    // store: show stored reg as primary
    snprintf(out, n, "%s %s", mn, reg_name(m->rs2));
    return;
  }

  if ((m->flags & F_HAS_RS1) && (m->flags & F_HAS_RS2)) {
    snprintf(out, n, "%s %s %s %s", mn, reg_name(m->rd), reg_name(m->rs1), reg_name(m->rs2));
    return;
  }
  if (m->flags & F_HAS_RS1) {
    snprintf(out, n, "%s %s %s", mn, reg_name(m->rd), reg_name(m->rs1));
    return;
  }
  snprintf(out, n, "%s", mn);
}

// -----------------------------
// Runtime analysis helper
// -----------------------------

__attribute__((noinline)) static void lej_analyze(const ins_meta_t *m) {
  thread_state_t *ts = tls;
  if (ts == NULL || m == NULL) return;

  const uint64_t cur_seq = ++ts->dyn_seq;

  ts->total_insts++;

  const bool is_load  = (m->flags & F_IS_LOAD) != 0;
  const bool is_store = (m->flags & F_IS_STORE) != 0;
  const bool is_long  = is_load || is_store;
  const bool is_exp   = (m->flags & F_IS_EXPENSIVE) != 0;

  if (is_long) ts->long_insts++;
  if (is_exp)  ts->expensive_insts++;

  // JOIN detection: must read two regs and be join-candidate
  if ((m->flags & F_IS_JOIN_CAND) && (m->flags & F_HAS_RS1) && (m->flags & F_HAS_RS2)) {
    const uint8_t rs1 = m->rs1;
    const uint8_t rs2 = m->rs2;

    if (rs1 < 32 && rs2 < 32) {
      producer_t p1 = ts->last_prod[rs1];
      producer_t p2 = ts->last_prod[rs2];

      const bool p1L = (p1.kind == PROD_LONG_LOAD);
      const bool p2L = (p2.kind == PROD_LONG_LOAD);
      const bool p1E = (p1.kind == PROD_EXPENSIVE);
      const bool p2E = (p2.kind == PROD_EXPENSIVE);

      if ((p1L && p2E) || (p2L && p1E)) {
        const producer_t PL = p1L ? p1 : p2;
        const producer_t PE = p1E ? p1 : p2;

        // Basic ordering: both producers executed before join
        if (PL.seq < cur_seq && PE.seq < cur_seq && PL.meta && PE.meta) {
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

            // dyn-id = join's runtime sequence number
            chain_map_inc(&ts->chains, &k, cur_seq);
          }
        }
      }
    }
  }

  // Producer update if writes rd
  if ((m->flags & F_WRITES_RD) && m->rd != 0 && m->rd < 32) {
    producer_t *pd = &ts->last_prod[m->rd];

    prod_kind_t kind = PROD_OTHER;
    // L producer must be load (store doesn't produce a reg)
    if (is_load) {
      kind = PROD_LONG_LOAD;
    } else if (is_exp) {
      kind = PROD_EXPENSIVE;
    }

    pd->kind = kind;
    pd->seq  = cur_seq;
    pd->meta = m;
    pd->rd   = m->rd;
    pd->rs1  = (m->flags & F_HAS_RS1) ? m->rs1 : 0xFF;
    pd->rs2  = (m->flags & F_HAS_RS2) ? m->rs2 : 0xFF;
  }
}

// -----------------------------
// Plugin callbacks
// -----------------------------

static int lej_pre_thread(mambo_context *ctx);
static int lej_pre_inst(mambo_context *ctx);
static int lej_post_thread(mambo_context *ctx);
static int lej_exit(mambo_context *ctx);

__attribute__((constructor)) void lej_init_plugin(void) {
  mambo_context *ctx = mambo_register_plugin();
  assert(ctx != NULL);

  static bool inited = false;
  if (!inited) {
    inited = true;
    memset(&g, 0, sizeof(g));
    chain_map_init(&g.chains);
  }

  mambo_register_pre_thread_cb(ctx, &lej_pre_thread);
  mambo_register_pre_inst_cb(ctx, &lej_pre_inst);
  mambo_register_post_thread_cb(ctx, &lej_post_thread);
  mambo_register_exit_cb(ctx, &lej_exit);
}

static int lej_pre_thread(mambo_context *ctx) {
  thread_state_t *ts = (thread_state_t *)mambo_alloc(ctx, sizeof(thread_state_t));
  assert(ts != NULL);
  memset(ts, 0, sizeof(*ts));
  chain_map_init(&ts->chains);

  for (int i = 0; i < 32; i++) {
    ts->last_prod[i].kind = PROD_NONE;
    ts->last_prod[i].seq  = 0;
    ts->last_prod[i].meta = NULL;
    ts->last_prod[i].rd   = (uint8_t)i;
    ts->last_prod[i].rs1  = 0xFF;
    ts->last_prod[i].rs2  = 0xFF;
  }

  tls = ts;
  mambo_set_thread_plugin_data(ctx, ts);
  return 0;
}

static int lej_post_thread(mambo_context *ctx) {
  thread_state_t *ts = (thread_state_t *)mambo_get_thread_plugin_data(ctx);
  if (!ts) return 0;

  // Merge thread stats into global stats + global chains
  lock_acquire(&g.lock);

  g.total_insts     += ts->total_insts;
  g.long_insts      += ts->long_insts;
  g.expensive_insts += ts->expensive_insts;

  chain_map_merge(&g.chains, &ts->chains);

  lock_release(&g.lock);

  mambo_free(ctx, ts);
  if (tls == ts) tls = NULL;
  return 0;
}

static void dump_chains(FILE *fp, const chain_map_t *m) {
  uint64_t unique = 0;
  uint64_t total  = 0;

  for (uint32_t i = 0; i < LEJ_BUCKETS; i++) {
    for (const chain_entry_t *e = m->buckets[i]; e; e = e->next) {
      unique++;
      total += e->count;

      // Build printable strings using the stored key (PC-based), but we still want the
      // exact operand format. We'll reconstruct pseudo-ins_meta from the key fields.
      ins_meta_t Lm = { .pc = e->key.pcL, .inst = e->key.instL, .rd = e->key.rdL, .rs1 = e->key.rs1L, .rs2 = e->key.rs2L, .flags = F_IS_LOAD | F_WRITES_RD | F_HAS_RS1 };
      ins_meta_t Em = { .pc = e->key.pcE, .inst = e->key.instE, .rd = e->key.rdE, .rs1 = e->key.rs1E, .rs2 = e->key.rs2E, .flags = F_WRITES_RD | F_HAS_RS1 | F_HAS_RS2 };
      ins_meta_t Jm = { .pc = e->key.pcJ, .inst = e->key.instJ, .rd = e->key.rdJ, .rs1 = e->key.rs1J, .rs2 = e->key.rs2J, .flags = F_WRITES_RD | F_HAS_RS1 | F_HAS_RS2 | F_IS_JOIN_CAND };

      char sL[96], sE[96], sJ[96];
      fmt_ins(sL, sizeof(sL), &Lm);
      fmt_ins(sE, sizeof(sE), &Em);
      fmt_ins(sJ, sizeof(sJ), &Jm);

      fprintf(fp,
              "dyn_id(first_seen): %" PRIu64 " | count: %" PRIu64 "\n",
              e->first_dyn_id, e->count);

      fprintf(fp, "consumer: %s (pc=0x%" PRIxPTR ")\n", sJ, (uintptr_t)e->key.pcJ);
      fprintf(fp, "producer: %s (%s reg) (pc=0x%" PRIxPTR ")\n", sE, reg_name(e->key.rdE), (uintptr_t)e->key.pcE);
      fprintf(fp, "producer: %s (%s reg) (pc=0x%" PRIxPTR ")\n", sL, reg_name(e->key.rdL), (uintptr_t)e->key.pcL);

      fprintf(fp, "chain:\n%s\n%s\n%s\n\n", sE, sL, sJ);
    }
  }

  fprintf(fp, "Unique chains: %" PRIu64 "\n", unique);
  fprintf(fp, "Total LEJ detections (sum of counts): %" PRIu64 "\n", total);
}

static int lej_exit(mambo_context *ctx) {
  (void)ctx;

  FILE *fp = fopen("chain.txt", "w");
  if (!fp) fp = stderr;

  const uint64_t total_det = chain_map_total_detections(&g.chains);
  const uint64_t unique    = chain_map_unique(&g.chains);

  fprintf(fp, "==== GLOBAL STATS (DYNAMIC) ====\n");
  fprintf(fp, "total_instructions_executed: %" PRIu64 "\n", g.total_insts);
  fprintf(fp, "total_long_executed (load+store): %" PRIu64 "\n", g.long_insts);
  fprintf(fp, "total_expensive_executed (mul/div/rem): %" PRIu64 "\n", g.expensive_insts);
  fprintf(fp, "total_lej_detections: %" PRIu64 "\n", total_det);
  fprintf(fp, "unique_lej_chains: %" PRIu64 "\n\n", unique);

  fprintf(fp, "==== UNIQUE LEJ CHAINS ====\n");
  dump_chains(fp, &g.chains);

  if (fp != stderr) fclose(fp);
  return 0;
}

static int lej_pre_inst(mambo_context *ctx) {
  // Allocate per-instruction meta (lives for the lifetime of translated code)
  ins_meta_t *m = (ins_meta_t *)mambo_alloc(ctx, sizeof(ins_meta_t));
  assert(m != NULL);
  memset(m, 0, sizeof(*m));

  m->pc   = (uintptr_t)ctx->code.read_address;
  m->inst = (uint16_t)ctx->code.inst;

  // Raw encoding (best-effort)
  m->raw32 = load_u32(ctx->code.read_address);

  uint8_t flags = 0;
  if (mambo_is_load(ctx))  flags |= F_IS_LOAD;
  if (mambo_is_store(ctx)) flags |= F_IS_STORE;
  if (is_expensive_enum(m->inst)) flags |= F_IS_EXPENSIVE;

  decode_rv_regs_and_kind(m->raw32, &m->rd, &m->rs1, &m->rs2, &flags);
  m->flags = flags;

  // Instrumentation: runtime call (dynamic analysis)
  // This is what makes the whole thing dynamic.
  emit_safe_fcall_static_args(ctx, (void *)&lej_analyze, 1, (uintptr_t)m);

  return 0;
}

#endif // PLUGINS_NEW
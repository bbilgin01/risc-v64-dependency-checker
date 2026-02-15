#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include "../plugins.h"

typedef enum {
    REG_CLEAN,
    REG_PENDING_EXPENSIVE,
    REG_PENDING_LONG
} reg_status_t;

struct lej_thread_data {
    reg_status_t reg_file[32];
    uintptr_t last_producer_pc[32];
    uint64_t expensive_count;
    uint64_t long_count;
    uint64_t chain_count;
    bool in_roi;
};

#define RISCV_OPCODE(inst) (inst & 0x7F)
#define GET_RD(inst)       ((inst >> 7) & 0x1F)
#define GET_RS1(inst)      ((inst >> 15) & 0x1F)
#define GET_RS2(inst)      ((inst >> 20) & 0x1F)

int lej_pre_inst_handler(mambo_context *ctx) {
    struct lej_thread_data *data = mambo_get_thread_plugin_data(ctx);
    uint32_t inst = mambo_get_inst(ctx);
    uintptr_t pc = (uintptr_t)mambo_get_inst_addr(ctx);

    if (inst == 0x00100013) { data->in_roi = true; return 0; }
    if (inst == 0x00200013) { data->in_roi = false; return 0; }
    if (!data->in_roi) return 0;

    uint32_t op = RISCV_OPCODE(inst);
    uint32_t f3 = (inst >> 12) & 0x7;
    uint32_t f7 = (inst >> 25) & 0x7F;

    int rs1 = GET_RS1(inst);
    int rs2 = GET_RS2(inst);
    int rd  = GET_RD(inst);

    // --- 1. Consumer Analizi (Triple Chain Detection) ---
    if (rs1 != 0 && rs2 != 0) {
        bool match = false;
        if ((data->reg_file[rs1] == REG_PENDING_EXPENSIVE && data->reg_file[rs2] == REG_PENDING_LONG) ||
            (data->reg_file[rs1] == REG_PENDING_LONG && data->reg_file[rs2] == REG_PENDING_EXPENSIVE)) {
            
            data->chain_count++;
            extern FILE *chain_file;
            if (chain_file) {
                fprintf(chain_file, "INDEPENDENT_STALL [PC: 0x%" PRIxPTR "]: Consumer waits for unrelated P1(0x%" PRIxPTR ") and P2(0x%" PRIxPTR ")\n",
                        pc, data->last_producer_pc[rs1], data->last_producer_pc[rs2]);
            }
        }
    }

    // --- 2. Producer Analizi & Bağımsızlık Kontrolü ---
    if (rd != 0) {
        // Expensive: MUL/DIV
        if ((op == 0x33 || op == 0x3B) && f7 == 0x01) {
            // Kontrol: Eğer MUL'un RS1 veya RS2'si hali hazırda bir LONG bekliyorsa, 
            // bu bağımlı bir zincirdir. Biz "bağımsız" olanları arıyoruz.
            if (data->reg_file[rs1] != REG_PENDING_LONG && data->reg_file[rs2] != REG_PENDING_LONG) {
                data->expensive_count++;
                data->reg_file[rd] = REG_PENDING_EXPENSIVE;
                data->last_producer_pc[rd] = pc;
            }
        } 
        // Long: LOAD
        else if (op == 0x03) {
            // Kontrol: Eğer LOAD'un adres register'ı (rs1) bir EXPENSIVE bekliyorsa, bağımlıdır.
            if (data->reg_file[rs1] != REG_PENDING_EXPENSIVE) {
                data->long_count++;
                data->reg_file[rd] = REG_PENDING_LONG;
                data->last_producer_pc[rd] = pc;
            }
        }
        // Store (0x23) rd yazmaz ama long_count artırırız
        else if (op == 0x23) {
             if (data->reg_file[rs1] != REG_PENDING_EXPENSIVE && data->reg_file[rs2] != REG_PENDING_EXPENSIVE) {
                data->long_count++;
             }
        }
        // Temizleme (rd yazan diğer her şey)
        else if (op != 0x63) {
            data->reg_file[rd] = REG_CLEAN;
        }
    }

    return 0;
}
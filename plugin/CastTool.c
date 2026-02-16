#ifdef PLUGINS_NEW

#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
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

FILE *chain_file = NULL;
uint64_t global_expensive = 0;
uint64_t global_long = 0;
uint64_t global_chains = 0;

#define RISCV_OPCODE(inst) (inst & 0x7F)
#define GET_RD(inst)       ((inst >> 7) & 0x1F)
#define GET_RS1(inst)      ((inst >> 15) & 0x1F)
#define GET_RS2(inst)      ((inst >> 20) & 0x1F)

int lej_pre_thread_handler(mambo_context *ctx) {
    struct lej_thread_data *data = mambo_alloc(ctx, sizeof(struct lej_thread_data));
    assert(data != NULL);
    
    for (int i = 0; i < 32; i++) {
        data->reg_file[i] = REG_CLEAN;
        data->last_producer_pc[i] = 0;
    }
    data->expensive_count = 0;
    data->long_count = 0;
    data->chain_count = 0;
    data->in_roi = false;

    mambo_set_thread_plugin_data(ctx, data);
    return 0;
}

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

    if (rs1 != 0 && rs2 != 0) {
        if ((data->reg_file[rs1] == REG_PENDING_EXPENSIVE && data->reg_file[rs2] == REG_PENDING_LONG) ||
            (data->reg_file[rs1] == REG_PENDING_LONG && data->reg_file[rs2] == REG_PENDING_EXPENSIVE)) {
            
            data->chain_count++;
            if (chain_file) {
                fprintf(chain_file, "INDEPENDENT_STALL [PC: 0x%" PRIxPTR "]: RS1_PC:0x%" PRIxPTR " RS2_PC:0x%" PRIxPTR "\n",
                        pc, data->last_producer_pc[rs1], data->last_producer_pc[rs2]);
            }
        }
    }

    if (rd != 0) {
        if ((op == 0x33 || op == 0x3B) && f7 == 0x01) { 
            if (data->reg_file[rs1] != REG_PENDING_LONG && data->reg_file[rs2] != REG_PENDING_LONG) {
                data->expensive_count++;
                data->reg_file[rd] = REG_PENDING_EXPENSIVE;
                data->last_producer_pc[rd] = pc;
            }
        } else if (op == 0x03) { 
            if (data->reg_file[rs1] != REG_PENDING_EXPENSIVE) {
                data->long_count++;
                data->reg_file[rd] = REG_PENDING_LONG;
                data->last_producer_pc[rd] = pc;
            }
        } else if (op == 0x23) { 
             if (data->reg_file[rs1] != REG_PENDING_EXPENSIVE && data->reg_file[rs2] != REG_PENDING_EXPENSIVE) {
                data->long_count++;
             }
        } else if (op != 0x63) { 
            data->reg_file[rd] = REG_CLEAN;
        }
    }
    return 0;
}

int lej_post_thread_handler(mambo_context *ctx) {
    struct lej_thread_data *data = mambo_get_thread_plugin_data(ctx);
    
    atomic_increment_u64(&global_expensive, data->expensive_count);
    atomic_increment_u64(&global_long, data->long_count);
    atomic_increment_u64(&global_chains, data->chain_count);

    mambo_free(ctx, data);
    return 0;
}

int lej_exit_handler(mambo_context *ctx) {
    FILE *stat_file = fopen("stat.txt", "w");
    if (stat_file) {
        fprintf(stat_file, "Total Expensive: %lu\n", global_expensive);
        fprintf(stat_file, "Total Long: %lu\n", global_long);
        fprintf(stat_file, "Independent Stall Chains: %lu\n", global_chains);
        fclose(stat_file);
    }
    if (chain_file) fclose(chain_file);
    return 0;
}

__attribute__((constructor)) void lej_init_plugin() {
    mambo_context *ctx = mambo_register_plugin();
    assert(ctx != NULL);

    chain_file = fopen("chain.txt", "w");

    mambo_register_pre_thread_cb(ctx, &lej_pre_thread_handler);
    mambo_register_pre_inst_cb(ctx, &lej_pre_inst_handler);
    mambo_register_post_thread_cb(ctx, &lej_post_thread_handler);
    mambo_register_exit_cb(ctx, &lej_exit_handler);
}

#endif
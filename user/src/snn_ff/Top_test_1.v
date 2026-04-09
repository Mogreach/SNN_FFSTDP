`timescale 1ns / 1ps
`include "Top_test_cfg.vh"
//****************************************VSCODE PLUG-IN**********************************//
//----------------------------------------------------------------------------------------//
// IDE :                   VSCODE
// VSCODE plug-in version: Verilog-Hdl-Format-3.5.20250220
// VSCODE plug-in author : Jiang Percy
//----------------------------------------------------------------------------------------//
//****************************************Copyright (c)***********************************//
// Copyright(C)            Personal
// All rights reserved
// File name:
// Last modified Date:     2025/02/26 16:10:53
// Last Version:           V1.0
// Descriptions:
//----------------------------------------------------------------------------------------//
// Created by:             Sephiroth
// Created date:           2025/02/26 16:10:53
// mail      :             1245598043@qq.com
// Version:                V1.0
// TEXT NAME:              Top_test.v
// PATH:                   D:\MyProject\FPGA_prj\SNN_FFSTBP\rtl\snn_ff\Top_test.v
// Descriptions:
//   Pure-Verilog top wrapper for multi-layer ODIN_ffstdp chaining.
//   It is intended for Vivado Block Design / Module Reference flows.
//
//   Example 784-256-256:
//     .LAYER_NUM(2)
//     .LAYER_OUTPUT_NEURON_CFG(`TOP_TEST_CFG2(16'd256, 16'd256))
//     .LAYER_POST_NEUR_PARALLEL_CFG(`TOP_TEST_CFG2(16'd8, 16'd8))
//
//   Notes:
//   1. LAYER_NUM must be <= `TOP_TEST_MAX_LAYERS.
//   2. Each config item uses `TOP_TEST_CFG_ITEM_WIDTH bits.
//   3. Item 0 is the first hidden/output layer after INPUT_NEURON.
//----------------------------------------------------------------------------------------//
//****************************************************************************************//

module Top_test_1
#(
    parameter integer TIME_STEP = 8,
    parameter integer INPUT_NEURON = 784,
    parameter integer OUTPUT_NEURON = 512,
    parameter integer AER_IN_CORE_WIDTH = 12,

    parameter integer POST_NEUR_PARALLEL = 8,
    parameter integer POST_NEUR_DATA_WIDTH = 20,
    parameter integer POST_NEUR_MEM_WIDTH = 13,
    parameter integer POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter integer WEIGHT_WIDTH = 9,
    parameter integer GRAD_WIDTH = 9,
    parameter integer GOODNESS_WIDTH = 20,

    parameter integer LAYER_NUM = 3,
    parameter [`TOP_TEST_CFG_WIDTH-1:0] LAYER_OUTPUT_NEURON_CFG = {16'd512,16'd512,16'd512},
    parameter [`TOP_TEST_CFG_WIDTH-1:0] LAYER_POST_NEUR_PARALLEL_CFG = {16'd8,16'd8,16'd8}
)
(
    input  wire        CLK,
    input  wire        RST,
    input  wire [11:0] AERIN_ADDR,
    input  wire        AERIN_REQ,
    input  wire        IS_POS,
    input  wire        IS_TRAIN,
    output wire        AERIN_ACK,
    output reg  [31:0] GOODNESS,
    output wire        PROCESS_DONE
);

    localparam integer TOP_GOODNESS_ACC_WIDTH = 32 + $clog2(`TOP_TEST_MAX_LAYERS + 1);
    localparam integer MAX_AER_WIDTH = calc_max_aer_width(0);

    wire [`TOP_TEST_MAX_LAYERS-1:0] layer_aerin_ack;
    wire [`TOP_TEST_MAX_LAYERS-1:0] layer_aerout_req;
    wire [`TOP_TEST_MAX_LAYERS-1:0] layer_aerout_ack;
    wire [`TOP_TEST_MAX_LAYERS-1:0] layer_done;
    wire [`TOP_TEST_MAX_LAYERS*MAX_AER_WIDTH-1:0] layer_aerout_addr_bus;
    wire [`TOP_TEST_MAX_LAYERS*32-1:0] layer_goodness_bus;

    wire final_aerout_req;
    wire final_aerout_ack;
    reg  final_aerout_ack_reg;
    reg  [5:0] final_aerout_ack_delay;

    reg  [TOP_GOODNESS_ACC_WIDTH-1:0] goodness_accum;
    reg  [TOP_GOODNESS_ACC_WIDTH-1:0] goodness_done_sum;
    reg  any_layer_done;
    reg  clear_goodness_pending;

    integer i;

    function integer cfg_item;
        input [`TOP_TEST_CFG_WIDTH-1:0] cfg_bus;
        input integer idx;
        reg [`TOP_TEST_CFG_WIDTH-1:0] shifted_bus;
        begin
            shifted_bus = cfg_bus >> (idx * `TOP_TEST_CFG_ITEM_WIDTH);
            cfg_item = shifted_bus[`TOP_TEST_CFG_ITEM_WIDTH-1:0];
        end
    endfunction

    function integer layer_output_neuron;
        input integer idx;
        integer value;
        begin
            value = cfg_item(LAYER_OUTPUT_NEURON_CFG, idx);
            if ((idx < 0) || (idx >= LAYER_NUM) || (value <= 0))
                layer_output_neuron = OUTPUT_NEURON;
            else
                layer_output_neuron = value;
        end
    endfunction

    function integer layer_post_neur_parallel;
        input integer idx;
        integer value;
        begin
            value = cfg_item(LAYER_POST_NEUR_PARALLEL_CFG, idx);
            if ((idx < 0) || (idx >= LAYER_NUM) || (value <= 0))
                layer_post_neur_parallel = POST_NEUR_PARALLEL;
            else
                layer_post_neur_parallel = value;
        end
    endfunction

    function integer calc_max_aer_width;
        input integer dummy;
        integer idx;
        integer cur_width;
        begin
            calc_max_aer_width = AER_IN_CORE_WIDTH;
            for (idx = 0; idx < `TOP_TEST_MAX_LAYERS; idx = idx + 1) begin
                if (idx < LAYER_NUM) begin
                    cur_width = 2 + $clog2(layer_output_neuron(idx));
                    if (cur_width > calc_max_aer_width)
                        calc_max_aer_width = cur_width;
                end
            end
        end
    endfunction

    function [31:0] sat_to32;
        input [TOP_GOODNESS_ACC_WIDTH-1:0] value;
        begin
            if (|value[TOP_GOODNESS_ACC_WIDTH-1:32])
                sat_to32 = 32'hFFFF_FFFF;
            else
                sat_to32 = value[31:0];
        end
    endfunction

    assign AERIN_ACK = (LAYER_NUM > 0) ? layer_aerin_ack[0] : 1'b0;
    assign PROCESS_DONE = (LAYER_NUM > 0) ? layer_done[LAYER_NUM-1] : 1'b0;
    assign final_aerout_req = (LAYER_NUM > 0) ? layer_aerout_req[LAYER_NUM-1] : 1'b0;
    assign final_aerout_ack = final_aerout_ack_delay[5];

    always @(*) begin
        goodness_done_sum = {TOP_GOODNESS_ACC_WIDTH{1'b0}};
        any_layer_done = 1'b0;
        for (i = 0; i < `TOP_TEST_MAX_LAYERS; i = i + 1) begin
            if ((i < LAYER_NUM) && layer_done[i]) begin
                any_layer_done = 1'b1;
                goodness_done_sum = goodness_done_sum + {{(TOP_GOODNESS_ACC_WIDTH-32){1'b0}}, layer_goodness_bus[i*32 +: 32]};
            end
        end
    end

    always @(posedge CLK or posedge RST) begin
        if (RST)
            final_aerout_ack_reg <= 1'b0;
        else if (final_aerout_req)
            final_aerout_ack_reg <= 1'b1;
        else
            final_aerout_ack_reg <= 1'b0;
    end

    always @(posedge CLK or posedge RST) begin
        if (RST)
            final_aerout_ack_delay <= 6'b0;
        else
            final_aerout_ack_delay <= {final_aerout_ack_delay[4:0], final_aerout_ack_reg};
    end

    // GOODNESS semantics:
    // 1. Each layer contributes once when its ONE_SAMPLE_FINISH pulses.
    // 2. Top GOODNESS holds the accumulated sum across finished layers.
    // 3. After the last layer finishes, GOODNESS is cleared on the next cycle.
    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            goodness_accum <= {TOP_GOODNESS_ACC_WIDTH{1'b0}};
            GOODNESS <= 32'd0;
            clear_goodness_pending <= 1'b0;
        end else if (clear_goodness_pending) begin
            goodness_accum <= {TOP_GOODNESS_ACC_WIDTH{1'b0}};
            GOODNESS <= 32'd0;
            clear_goodness_pending <= 1'b0;
        end else begin
            if (any_layer_done) begin
                goodness_accum <= goodness_accum + goodness_done_sum;
                GOODNESS <= sat_to32(goodness_accum + goodness_done_sum);
            end else begin
                GOODNESS <= sat_to32(goodness_accum);
            end

            if ((LAYER_NUM > 0) && layer_done[LAYER_NUM-1])
                clear_goodness_pending <= 1'b1;
            else
                clear_goodness_pending <= 1'b0;
        end
    end

    genvar g;
    generate
        for (g = 0; g < `TOP_TEST_MAX_LAYERS; g = g + 1) begin : gen_layers
            localparam integer CUR_ACTIVE = (g < LAYER_NUM) ? 1 : 0;
            localparam integer CUR_INPUT_NEURON = (g == 0) ? INPUT_NEURON : layer_output_neuron(g-1);
            localparam integer CUR_OUTPUT_NEURON = layer_output_neuron(g);
            localparam integer CUR_AER_IN_WIDTH = (g == 0) ? AER_IN_CORE_WIDTH : (2 + $clog2(layer_output_neuron(g-1)));
            localparam integer CUR_AER_OUT_WIDTH = 2 + $clog2(layer_output_neuron(g));
            localparam integer CUR_POST_NEUR_PARALLEL = layer_post_neur_parallel(g);

            if (CUR_ACTIVE) begin : gen_active_layer
                wire [CUR_AER_IN_WIDTH-1:0]  cur_aerin_addr;
                wire                         cur_aerin_req;
                wire                         cur_aerin_ack;
                wire [CUR_AER_OUT_WIDTH-1:0] cur_aerout_addr;
                wire                         cur_aerout_req;
                wire [31:0]                  cur_goodness;
                wire                         cur_done;

                if (g == 0) begin : gen_first_layer_input
                    assign cur_aerin_addr = AERIN_ADDR[CUR_AER_IN_WIDTH-1:0];
                    assign cur_aerin_req = AERIN_REQ;
                end else begin : gen_chain_layer_input
                    assign cur_aerin_addr = layer_aerout_addr_bus[(g-1)*MAX_AER_WIDTH +: CUR_AER_IN_WIDTH];
                    assign cur_aerin_req = layer_aerout_req[g-1];
                end

                if (g == (LAYER_NUM-1)) begin : gen_last_layer_ack
                    assign layer_aerout_ack[g] = final_aerout_ack;
                end else begin : gen_mid_layer_ack
                    assign layer_aerout_ack[g] = layer_aerin_ack[g+1];
                end

                assign layer_aerin_ack[g] = cur_aerin_ack;
                assign layer_aerout_req[g] = cur_aerout_req;
                assign layer_aerout_addr_bus[g*MAX_AER_WIDTH +: MAX_AER_WIDTH] = {{(MAX_AER_WIDTH-CUR_AER_OUT_WIDTH){1'b0}}, cur_aerout_addr};
                assign layer_goodness_bus[g*32 +: 32] = cur_goodness;
                assign layer_done[g] = cur_done;

                Top_test_odin_layer #(
                    .TIME_STEP                 (TIME_STEP),
                    .INPUT_NEURON              (CUR_INPUT_NEURON),
                    .OUTPUT_NEURON             (CUR_OUTPUT_NEURON),
                    .AER_IN_CORE_WIDTH         (CUR_AER_IN_WIDTH),
                    .POST_NEUR_PARALLEL        (CUR_POST_NEUR_PARALLEL),
                    .POST_NEUR_DATA_WIDTH      (POST_NEUR_DATA_WIDTH),
                    .POST_NEUR_MEM_WIDTH       (POST_NEUR_MEM_WIDTH),
                    .POST_NEUR_SPIKE_CNT_WIDTH (POST_NEUR_SPIKE_CNT_WIDTH),
                    .WEIGHT_WIDTH              (WEIGHT_WIDTH),
                    .GRAD_WIDTH                (GRAD_WIDTH),
                    .GOODNESS_WIDTH            (GOODNESS_WIDTH)
                ) u_layer (
                    .CLK               (CLK),
                    .RST               (RST),
                    .AERIN_ADDR        (cur_aerin_addr),
                    .AERIN_REQ         (cur_aerin_req),
                    .IS_POS            (IS_POS),
                    .IS_TRAIN          (IS_TRAIN),
                    .AERIN_ACK         (cur_aerin_ack),
                    .AEROUT_ADDR       (cur_aerout_addr),
                    .AEROUT_REQ        (cur_aerout_req),
                    .AEROUT_ACK        (layer_aerout_ack[g]),
                    .GOODNESS          (cur_goodness),
                    .ONE_SAMPLE_FINISH (cur_done)
                );
            end else begin : gen_inactive_layer
                assign layer_aerin_ack[g] = 1'b0;
                assign layer_aerout_req[g] = 1'b0;
                assign layer_aerout_ack[g] = 1'b0;
                assign layer_aerout_addr_bus[g*MAX_AER_WIDTH +: MAX_AER_WIDTH] = {MAX_AER_WIDTH{1'b0}};
                assign layer_goodness_bus[g*32 +: 32] = 32'd0;
                assign layer_done[g] = 1'b0;
            end
        end
    endgenerate

endmodule


module Top_test_odin_layer
#(
    parameter integer TIME_STEP = 8,
    parameter integer INPUT_NEURON = 784,
    parameter integer OUTPUT_NEURON = 256,
    parameter integer AER_IN_CORE_WIDTH = 12,

    parameter integer POST_NEUR_PARALLEL = 8,
    parameter integer POST_NEUR_DATA_WIDTH = 20,
    parameter integer POST_NEUR_MEM_WIDTH = 13,
    parameter integer POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter integer WEIGHT_WIDTH = 9,
    parameter integer GRAD_WIDTH = 9,
    parameter integer GOODNESS_WIDTH = 20
)
(
    input  wire                                 CLK,
    input  wire                                 RST,
    input  wire [AER_IN_CORE_WIDTH-1:0]         AERIN_ADDR,
    input  wire                                 AERIN_REQ,
    input  wire                                 IS_POS,
    input  wire                                 IS_TRAIN,
    output wire                                 AERIN_ACK,
    output wire [(2+$clog2(OUTPUT_NEURON))-1:0] AEROUT_ADDR,
    output wire                                 AEROUT_REQ,
    input  wire                                 AEROUT_ACK,
    output wire [31:0]                          GOODNESS,
    output wire                                 ONE_SAMPLE_FINISH
);

    localparam integer PRE_NEUR_ADDR_WIDTH = $clog2(INPUT_NEURON);
    localparam integer PRE_NEUR_WORD_ADDR_WIDTH = PRE_NEUR_ADDR_WIDTH;
    localparam integer PRE_NEUR_BYTE_ADDR_WIDTH = 0;
    localparam integer AER_OUT_CORE_WIDTH = 2 + $clog2(OUTPUT_NEURON);
    localparam integer POST_NEUR_ADDR_WIDTH = $clog2(OUTPUT_NEURON);
    localparam integer POST_NEUR_BYTE_ADDR_WIDTH = $clog2(POST_NEUR_PARALLEL);
    localparam integer POST_NEUR_WORD_ADDR_WIDTH = POST_NEUR_ADDR_WIDTH - POST_NEUR_BYTE_ADDR_WIDTH;
    localparam integer SYN_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH;
    localparam integer SYN_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL);
    localparam integer GRAD_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * GRAD_WIDTH;
    localparam integer GRAD_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL);
    localparam integer PRE_NEUR_DATA_WIDTH = TIME_STEP;

    wire [GOODNESS_WIDTH-1:0] avg_goodness;
    wire [POST_NEUR_MEM_WIDTH * POST_NEUR_PARALLEL -1:0] post_neur_mem_bus_unused;
    wire goodness_acc_valid_unused;
    wire goodness_clear_unused;
    wire [31:0] core_goodness;
    wire core_done;
    reg  [31:0] core_goodness_prev;
    reg  [31:0] core_goodness_latched;

    assign avg_goodness = {GOODNESS_WIDTH{1'b0}};
    assign GOODNESS = core_done ? core_goodness_prev : core_goodness_latched;

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            core_goodness_prev <= 32'd0;
            core_goodness_latched <= 32'd0;
        end else begin
            if (!core_done)
                core_goodness_prev <= core_goodness;
            if (core_done)
                core_goodness_latched <= core_goodness_prev;
        end
    end

    ODIN_ffstdp #(
        .TIME_STEP                 (TIME_STEP),
        .INPUT_NEURON              (INPUT_NEURON),
        .OUTPUT_NEURON             (OUTPUT_NEURON),
        .AER_IN_CORE_WIDTH         (AER_IN_CORE_WIDTH),
        .AER_OUT_CORE_WIDTH        (AER_OUT_CORE_WIDTH),
        .PRE_NEUR_ADDR_WIDTH       (PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH  (PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH  (PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH      (POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH (POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH (POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL        (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH       (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH      (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH       (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH (POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH      (SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH      (SYN_ARRAY_ADDR_WIDTH),
        .GRAD_ARRAY_DATA_WIDTH     (GRAD_ARRAY_DATA_WIDTH),
        .GRAD_ARRAY_ADDR_WIDTH     (GRAD_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH              (WEIGHT_WIDTH),
        .GRAD_WIDTH                (GRAD_WIDTH),
        .GOODNESS_WIDTH            (GOODNESS_WIDTH)
    ) u_ODIN_ffstdp (
        .CLK                (CLK),
        .RST                (RST),
        .IS_POS             (IS_POS),
        .IS_TRAIN           (IS_TRAIN),
        .AVG_GOODNESS       (avg_goodness),
        .AERIN_ADDR         (AERIN_ADDR),
        .AERIN_REQ          (AERIN_REQ),
        .AERIN_ACK          (AERIN_ACK),
        .AEROUT_ADDR        (AEROUT_ADDR),
        .AEROUT_REQ         (AEROUT_REQ),
        .AEROUT_ACK         (AEROUT_ACK),
        .ONE_SAMPLE_FINISH  (core_done),
        .GOODNESS           (core_goodness),
        .POST_NEUR_MEM_BUS  (post_neur_mem_bus_unused),
        .GOODNESS_ACC_VALID (goodness_acc_valid_unused),
        .GOODNESS_CLEAR     (goodness_clear_unused)
    );

    assign ONE_SAMPLE_FINISH = core_done;

endmodule

`timescale 1ns / 1ps
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
// TEXT NAME:              Top_test_1.v
// PATH:                   D:\MyProject\FPGA_prj\SNN_FFSTBP\rtl\snn_ff\Top_test_1.v
// Descriptions:
//   Hand-instantiated 784-512-512 top wrapper.
//   External ports stay aligned with the original Top_test style.
//----------------------------------------------------------------------------------------//
//****************************************************************************************//

module Top_test
#(
    parameter integer TIME_STEP = 8,
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

    localparam integer LAYER0_INPUT_NEURON = 784;
    localparam integer LAYER0_OUTPUT_NEURON = 512;
    localparam integer LAYER1_INPUT_NEURON = 512;
    localparam integer LAYER1_OUTPUT_NEURON = 512;

    localparam integer LAYER0_AER_IN_CORE_WIDTH = AER_IN_CORE_WIDTH;
    localparam integer LAYER0_AER_OUT_CORE_WIDTH = 2 + $clog2(LAYER0_OUTPUT_NEURON);
    localparam integer LAYER1_AER_IN_CORE_WIDTH = LAYER0_AER_OUT_CORE_WIDTH;
    localparam integer LAYER1_AER_OUT_CORE_WIDTH = 2 + $clog2(LAYER1_OUTPUT_NEURON);

    localparam integer PRE_NEUR_DATA_WIDTH = TIME_STEP;

    localparam integer LAYER0_PRE_NEUR_ADDR_WIDTH = $clog2(LAYER0_INPUT_NEURON);
    localparam integer LAYER0_PRE_NEUR_WORD_ADDR_WIDTH = LAYER0_PRE_NEUR_ADDR_WIDTH;
    localparam integer LAYER0_PRE_NEUR_BYTE_ADDR_WIDTH = 0;
    localparam integer LAYER0_POST_NEUR_ADDR_WIDTH = $clog2(LAYER0_OUTPUT_NEURON);
    localparam integer LAYER0_POST_NEUR_BYTE_ADDR_WIDTH = $clog2(POST_NEUR_PARALLEL);
    localparam integer LAYER0_POST_NEUR_WORD_ADDR_WIDTH = LAYER0_POST_NEUR_ADDR_WIDTH - LAYER0_POST_NEUR_BYTE_ADDR_WIDTH;
    localparam integer LAYER0_SYN_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH;
    localparam integer LAYER0_SYN_ARRAY_ADDR_WIDTH = $clog2(LAYER0_INPUT_NEURON * LAYER0_OUTPUT_NEURON / POST_NEUR_PARALLEL);
    localparam integer LAYER0_GRAD_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * GRAD_WIDTH;
    localparam integer LAYER0_GRAD_ARRAY_ADDR_WIDTH = $clog2(LAYER0_INPUT_NEURON * LAYER0_OUTPUT_NEURON / POST_NEUR_PARALLEL);

    localparam integer LAYER1_PRE_NEUR_ADDR_WIDTH = $clog2(LAYER1_INPUT_NEURON);
    localparam integer LAYER1_PRE_NEUR_WORD_ADDR_WIDTH = LAYER1_PRE_NEUR_ADDR_WIDTH;
    localparam integer LAYER1_PRE_NEUR_BYTE_ADDR_WIDTH = 0;
    localparam integer LAYER1_POST_NEUR_ADDR_WIDTH = $clog2(LAYER1_OUTPUT_NEURON);
    localparam integer LAYER1_POST_NEUR_BYTE_ADDR_WIDTH = $clog2(POST_NEUR_PARALLEL);
    localparam integer LAYER1_POST_NEUR_WORD_ADDR_WIDTH = LAYER1_POST_NEUR_ADDR_WIDTH - LAYER1_POST_NEUR_BYTE_ADDR_WIDTH;
    localparam integer LAYER1_SYN_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH;
    localparam integer LAYER1_SYN_ARRAY_ADDR_WIDTH = $clog2(LAYER1_INPUT_NEURON * LAYER1_OUTPUT_NEURON / POST_NEUR_PARALLEL);
    localparam integer LAYER1_GRAD_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * GRAD_WIDTH;
    localparam integer LAYER1_GRAD_ARRAY_ADDR_WIDTH = $clog2(LAYER1_INPUT_NEURON * LAYER1_OUTPUT_NEURON / POST_NEUR_PARALLEL);

    wire                               layer0_aerin_ack;
    wire [LAYER0_AER_OUT_CORE_WIDTH-1:0] layer0_aerout_addr;
    wire                               layer0_aerout_req;
    wire                               layer0_aerout_ack;
    wire [31:0]                        layer0_goodness_raw;
    wire [31:0]                        layer0_goodness;
    wire                               layer0_done_raw;
    wire                               layer0_done;
    reg  [31:0]                        layer0_goodness_prev;
    reg  [31:0]                        layer0_goodness_latched;
    wire [POST_NEUR_MEM_WIDTH * POST_NEUR_PARALLEL -1:0] layer0_post_neur_mem_bus_unused;
    wire                               layer0_goodness_acc_valid_unused;
    wire                               layer0_goodness_clear_unused;
    wire [GOODNESS_WIDTH-1:0]          layer0_avg_goodness;

    wire                               layer1_aerin_ack;
    wire [LAYER1_AER_OUT_CORE_WIDTH-1:0] layer1_aerout_addr;
    wire                               layer1_aerout_req;
    wire                               layer1_aerout_ack;
    wire [31:0]                        layer1_goodness_raw;
    wire [31:0]                        layer1_goodness;
    wire                               layer1_done_raw;
    wire                               layer1_done;
    reg  [31:0]                        layer1_goodness_prev;
    reg  [31:0]                        layer1_goodness_latched;
    wire [POST_NEUR_MEM_WIDTH * POST_NEUR_PARALLEL -1:0] layer1_post_neur_mem_bus_unused;
    wire                               layer1_goodness_acc_valid_unused;
    wire                               layer1_goodness_clear_unused;
    wire [GOODNESS_WIDTH-1:0]          layer1_avg_goodness;

    reg                                final_aerout_ack_reg;
    reg  [5:0]                         final_aerout_ack_delay;
    wire                               final_aerout_ack;

    reg  [33:0]                        goodness_accum;
    reg                                clear_goodness_pending;
    wire [33:0]                        goodness_done_sum;
    wire                               any_layer_done;

    function [31:0] sat34_to32;
        input [33:0] value;
        begin
            if (|value[33:32])
                sat34_to32 = 32'hFFFF_FFFF;
            else
                sat34_to32 = value[31:0];
        end
    endfunction

    assign layer0_avg_goodness = {GOODNESS_WIDTH{1'b0}};
    assign layer1_avg_goodness = {GOODNESS_WIDTH{1'b0}};

    assign AERIN_ACK = layer0_aerin_ack;
    assign PROCESS_DONE = layer1_done;

    assign layer0_aerout_ack = layer1_aerin_ack;
    assign final_aerout_ack = final_aerout_ack_delay[5];
    assign layer1_aerout_ack = final_aerout_ack;

    assign layer0_done = layer0_done_raw;
    assign layer1_done = layer1_done_raw;
    assign layer0_goodness = layer0_done_raw ? layer0_goodness_prev : layer0_goodness_latched;
    assign layer1_goodness = layer1_done_raw ? layer1_goodness_prev : layer1_goodness_latched;

    assign any_layer_done = layer0_done | layer1_done;
    assign goodness_done_sum = (layer0_done ? {2'b00, layer0_goodness} : 34'd0) +
                               (layer1_done ? {2'b00, layer1_goodness} : 34'd0);

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            layer0_goodness_prev <= 32'd0;
            layer0_goodness_latched <= 32'd0;
            layer1_goodness_prev <= 32'd0;
            layer1_goodness_latched <= 32'd0;
        end else begin
            if (!layer0_done_raw)
                layer0_goodness_prev <= layer0_goodness_raw;
            if (layer0_done_raw)
                layer0_goodness_latched <= layer0_goodness_prev;

            if (!layer1_done_raw)
                layer1_goodness_prev <= layer1_goodness_raw;
            if (layer1_done_raw)
                layer1_goodness_latched <= layer1_goodness_prev;
        end
    end

    always @(posedge CLK or posedge RST) begin
        if (RST)
            final_aerout_ack_reg <= 1'b0;
        else if (layer1_aerout_req)
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
    // 3. After the second layer finishes, GOODNESS is cleared on the next cycle.
    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            goodness_accum <= 34'd0;
            GOODNESS <= 32'd0;
            clear_goodness_pending <= 1'b0;
        end else if (clear_goodness_pending) begin
            goodness_accum <= 34'd0;
            GOODNESS <= 32'd0;
            clear_goodness_pending <= 1'b0;
        end else begin
            if (any_layer_done) begin
                goodness_accum <= goodness_accum + goodness_done_sum;
                GOODNESS <= sat34_to32(goodness_accum + goodness_done_sum);
            end else begin
                GOODNESS <= sat34_to32(goodness_accum);
            end

            if (layer1_done)
                clear_goodness_pending <= 1'b1;
            else
                clear_goodness_pending <= 1'b0;
        end
    end

    ODIN_ffstdp #(
        .TIME_STEP                 (TIME_STEP),
        .INPUT_NEURON              (LAYER0_INPUT_NEURON),
        .OUTPUT_NEURON             (LAYER0_OUTPUT_NEURON),
        .AER_IN_CORE_WIDTH         (LAYER0_AER_IN_CORE_WIDTH),
        .AER_OUT_CORE_WIDTH        (LAYER0_AER_OUT_CORE_WIDTH),
        .PRE_NEUR_ADDR_WIDTH       (LAYER0_PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH  (LAYER0_PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH  (LAYER0_PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH      (LAYER0_POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH (LAYER0_POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH (LAYER0_POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL        (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH       (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH      (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH       (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH (POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH      (LAYER0_SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH      (LAYER0_SYN_ARRAY_ADDR_WIDTH),
        .GRAD_ARRAY_DATA_WIDTH     (LAYER0_GRAD_ARRAY_DATA_WIDTH),
        .GRAD_ARRAY_ADDR_WIDTH     (LAYER0_GRAD_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH              (WEIGHT_WIDTH),
        .GRAD_WIDTH                (GRAD_WIDTH),
        .GOODNESS_WIDTH            (GOODNESS_WIDTH)
    ) u_layer0 (
        .CLK                (CLK),
        .RST                (RST),
        .IS_POS             (IS_POS),
        .IS_TRAIN           (IS_TRAIN),
        .AVG_GOODNESS       (layer0_avg_goodness),
        .AERIN_ADDR         (AERIN_ADDR[LAYER0_AER_IN_CORE_WIDTH-1:0]),
        .AERIN_REQ          (AERIN_REQ),
        .AERIN_ACK          (layer0_aerin_ack),
        .AEROUT_ADDR        (layer0_aerout_addr),
        .AEROUT_REQ         (layer0_aerout_req),
        .AEROUT_ACK         (layer0_aerout_ack),
        .ONE_SAMPLE_FINISH  (layer0_done_raw),
        .GOODNESS           (layer0_goodness_raw),
        .POST_NEUR_MEM_BUS  (layer0_post_neur_mem_bus_unused),
        .GOODNESS_ACC_VALID (layer0_goodness_acc_valid_unused),
        .GOODNESS_CLEAR     (layer0_goodness_clear_unused)
    );

    ODIN_ffstdp #(
        .TIME_STEP                 (TIME_STEP),
        .INPUT_NEURON              (LAYER1_INPUT_NEURON),
        .OUTPUT_NEURON             (LAYER1_OUTPUT_NEURON),
        .AER_IN_CORE_WIDTH         (LAYER1_AER_IN_CORE_WIDTH),
        .AER_OUT_CORE_WIDTH        (LAYER1_AER_OUT_CORE_WIDTH),
        .PRE_NEUR_ADDR_WIDTH       (LAYER1_PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH  (LAYER1_PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH  (LAYER1_PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH      (LAYER1_POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH (LAYER1_POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH (LAYER1_POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL        (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH       (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH      (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH       (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH (POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH      (LAYER1_SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH      (LAYER1_SYN_ARRAY_ADDR_WIDTH),
        .GRAD_ARRAY_DATA_WIDTH     (LAYER1_GRAD_ARRAY_DATA_WIDTH),
        .GRAD_ARRAY_ADDR_WIDTH     (LAYER1_GRAD_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH              (WEIGHT_WIDTH),
        .GRAD_WIDTH                (GRAD_WIDTH),
        .GOODNESS_WIDTH            (GOODNESS_WIDTH)
    ) u_layer1 (
        .CLK                (CLK),
        .RST                (RST),
        .IS_POS             (IS_POS),
        .IS_TRAIN           (IS_TRAIN),
        .AVG_GOODNESS       (layer1_avg_goodness),
        .AERIN_ADDR         (layer0_aerout_addr),
        .AERIN_REQ          (layer0_aerout_req),
        .AERIN_ACK          (layer1_aerin_ack),
        .AEROUT_ADDR        (layer1_aerout_addr),
        .AEROUT_REQ         (layer1_aerout_req),
        .AEROUT_ACK         (layer1_aerout_ack),
        .ONE_SAMPLE_FINISH  (layer1_done_raw),
        .GOODNESS           (layer1_goodness_raw),
        .POST_NEUR_MEM_BUS  (layer1_post_neur_mem_bus_unused),
        .GOODNESS_ACC_VALID (layer1_goodness_acc_valid_unused),
        .GOODNESS_CLEAR     (layer1_goodness_clear_unused)
    );

endmodule

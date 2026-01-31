`timescale 1ns / 1ps
//****************************************VSCODE PLUG-IN**********************************//
//----------------------------------------------------------------------------------------
// IDE :                   VSCODE     
// VSCODE plug-in version: Verilog-Hdl-Format-3.5.20250220
// VSCODE plug-in author : Jiang Percy
//----------------------------------------------------------------------------------------
//****************************************Copyright (c)***********************************//
// Copyright(C)            Personal
// All rights reserved     
// File name:              
// Last modified Date:     2025/02/26 16:12:19
// Last Version:           V1.0
// Descriptions:           
//----------------------------------------------------------------------------------------
// Created by:             Sephiroth
// Created date:           2025/02/26 16:12:19
// mail      :             1245598043@qq.com
// Version:                V1.0
// TEXT NAME:              ODIN_ffstdp.v
// PATH:                   D:\MyProject\FPGA_prj\SNN_FFSTBP\rtl\snn_ff\ODIN_ffstdp.v
// Descriptions:           
//                         
//----------------------------------------------------------------------------------------
//****************************************************************************************//
module ODIN_ffstdp #(
    parameter TIME_STEP = 8,
    parameter INPUT_NEURON = 784,
    parameter OUTPUT_NEURON = 256,
    parameter AER_IN_WIDTH = 12,
    parameter AER_OUT_WIDTH = 12,

    parameter PRE_NEUR_ADDR_WIDTH = 10,
    parameter PRE_NEUR_WORD_ADDR_WIDTH = 10,
    parameter PRE_NEUR_BYTE_ADDR_WIDTH = 0,

    parameter POST_NEUR_ADDR_WIDTH = 10,
    parameter POST_NEUR_WORD_ADDR_WIDTH = 8,
    parameter POST_NEUR_BYTE_ADDR_WIDTH = 2,
    parameter POST_NEUR_PARALLEL = 4,

    parameter PRE_NEUR_DATA_WIDTH = 8,
    parameter POST_NEUR_DATA_WIDTH = 32,
    parameter POST_NEUR_MEM_WIDTH = 12,
    parameter POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter SYN_ARRAY_DATA_WIDTH = 32,
    parameter SYN_ARRAY_ADDR_WIDTH = 16,
    parameter GRAD_ARRAY_DATA_WIDTH = 32,
    parameter GRAD_ARRAY_ADDR_WIDTH = 16,
    parameter WEIGHT_WIDTH = 8,
    parameter GRAD_WIDTH = 8,
    parameter GOODNESS_WIDTH = 20
) (
    // Global input     -------------------------------
    input  wire                 CLK,
    input  wire                 RST,
    input  wire                 IS_POS,      // 0: negative, 1: positive
    input  wire                 IS_TRAIN,    // 0: inference, 1: training
    input  wire        [GOODNESS_WIDTH-1:0] AVG_GOODNESS,
    // Input 12-bit AER -------------------------------
    input  wire [AER_IN_WIDTH-1:0] AERIN_ADDR,
    input  wire                 AERIN_REQ,
    output wire                 AERIN_ACK,

    // Output 10-bit AER -------------------------------
    output wire [AER_OUT_WIDTH-1:0] AEROUT_ADDR,
    output wire                 AEROUT_REQ,
    input  wire                 AEROUT_ACK,
    output wire                 ONE_SAMPLE_FINISH,
    output wire [         31:0] GOODNESS,
    // Connect to GOODNESS module
    output wire [POST_NEUR_MEM_WIDTH * POST_NEUR_PARALLEL -1:0] POST_NEUR_MEM_BUS, // 突触后神经元膜电位 
    output wire GOODNESS_ACC_VALID,
    output wire GOODNESS_CLEAR
);

    //----------------------------------------------------------------------------------
    //	Internal regs and wires
    //----------------------------------------------------------------------------------

    // Reset
    reg RST_sync_int, RST_sync;
    wire RSTN_sync;

    // AER output
    wire AEROUT_CTRL_BUSY;
    wire AEROUT_CTRL_FINISH;


    // SPI + parameter bank
    wire SPI_GATE_ACTIVITY, SPI_GATE_ACTIVITY_sync;
    wire SPI_OPEN_LOOP;
    wire SPI_AER_SRC_CTRL_nNEUR;
    wire [PRE_NEUR_ADDR_WIDTH-1:0] SPI_MAX_NEUR;

    // Controller
    wire CTRL_READBACK_EVENT;
    wire CTRL_PROG_EVENT;
    wire [2*PRE_NEUR_ADDR_WIDTH-1:0] CTRL_SPI_ADDR;
    wire [1:0] CTRL_OP_CODE;
    wire [2*PRE_NEUR_ADDR_WIDTH-1:0] CTRL_PROG_DATA;
    wire [SYN_ARRAY_ADDR_WIDTH-1:0] CTRL_SYNARRAY_ADDR;  // 控制信号: 突触数组地址
    wire CTRL_SYNARRAY_CS;  // 控制信号: 突触数组选择
    wire CTRL_SYNARRAY_WE;  // 控制信号: 突触数组写使能   
    wire CTRL_GRAD_ARRAY_CS; // 控制信号: 梯度数组选择
    wire CTRL_GRAD_ARRAY_WE; // 控制信号: 梯度数组写使能
    wire CTRL_SCHED_POP_N;
    wire [PRE_NEUR_ADDR_WIDTH-1:0] CTRL_SCHED_ADDR;
    wire CTRL_SCHED_EVENT_IN;
    wire [1:0] CTRL_SCHED_VIRTS;
    wire CTRL_AEROUT_POP_NEUR;
    wire CTRL_SYNA_WR_EVENT;  // 突触写事件
    wire CTRL_SYNA_RD_EVENT;  // 突触读事件
    wire [WEIGHT_WIDTH-1:0] CTRL_SYNA_PROG_DATA;  // 突触编程数据
    wire CTRL_NEUR_EVENT;  // 神经元事件
    wire CTRL_TSTEP_EVENT;  // 时间步事件
    wire CTRL_TREF_EVENT;  // 时间参考事件
    wire CTRL_WR_NEUR_EVENT;  // 写神经元事件
    wire CTRL_RD_NEUR_EVENT;  // 读神经元事件
    wire [POST_NEUR_DATA_WIDTH-1:0] CTRL_POST_NEUR_PROG_DATA;  // 神经元编程数据
    wire [PRE_NEUR_ADDR_WIDTH-1:0] CTRL_PRE_NEURON_ADDRESS;  // 预神经元地址
    wire [PRE_NEUR_ADDR_WIDTH-1:0] CTRL_POST_NEURON_ADDRESS;  // 后神经元地址
    wire CTRL_PRE_NEUR_CS;
    wire CTRL_PRE_NEUR_WE;
    wire CTRL_POST_NEUR_CS;
    wire CTRL_POST_NEUR_WE;
    wire CTRL_PRE_CNT_EN;  // 预神经元计数使能信号
    wire [$clog2(TIME_STEP)-1:0]CURRENT_TIME_STEP;
    wire CTRL_AEROUT_POP_TSTEP;
    wire CTRL_AEROUT_PUSH_NEUR;
    wire CTRL_AEROUT_TREF_FINISH;
    // Synaptic core
    wire [SYN_ARRAY_DATA_WIDTH-1:0] SYNARRAY_RDATA;  // 突触数组读数据

    // Scheduler
    wire SCHED_EMPTY;
    wire [AER_IN_WIDTH-1:0] SCHED_DATA_OUT;

    // Neuron core
    wire [POST_NEUR_DATA_WIDTH-1:0] NEUR_STATE;
    wire [POST_NEUR_PARALLEL-1:0] NEUR_EVENT_OUT;  // 神经元输出事件
    wire [PRE_NEUR_DATA_WIDTH-1:0] PRE_NEUR_S_CNT;  // 预神经元脉冲计数
    wire [POST_NEUR_SPIKE_CNT_WIDTH * POST_NEUR_PARALLEL -1 : 0] POST_NEUR_S_CNT;

    //----------------------------------------------------------------------------------
    //	Reset (with double sync barrier)
    //----------------------------------------------------------------------------------

    always @(posedge CLK) begin
        RST_sync_int <= RST;
        RST_sync     <= RST_sync_int;
    end

    assign RSTN_sync = ~RST_sync;

    //----------------------------------------------------------------------------------
    //	AER OUT
    //----------------------------------------------------------------------------------
    aer_outs #(
        .TIME_STEP                (TIME_STEP),
        .INPUT_NEURON             (INPUT_NEURON),
        .OUTPUT_NEURON            (OUTPUT_NEURON),
        .AER_OUT_WIDTH                (AER_OUT_WIDTH),
        .PRE_NEUR_ADDR_WIDTH      (PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH (PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH (PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH     (POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH(POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH(POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL       (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH      (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH     (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH      (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH(POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH     (SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH     (SYN_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH             (WEIGHT_WIDTH)
    ) u_aer_out (
        // Global input -----------------------------------
        .CLK                     (CLK),
        .RST                     (RST),
        // Neuron data inputs -----------------------------
        .NEUR_EVENT_OUT          (NEUR_EVENT_OUT),
        // Input from controller --------------------------
        .CTRL_TREF_EVENT         (CTRL_TREF_EVENT),
        .CTRL_POST_NEUR_CS       (CTRL_POST_NEUR_CS),
        .CTRL_POST_NEUR_WE       (CTRL_POST_NEUR_WE),
        .CTRL_AEROUT_PUSH_NEUR   (CTRL_AEROUT_PUSH_NEUR),
        .CTRL_AEROUT_POP_NEUR    (CTRL_AEROUT_POP_NEUR),
        .CTRL_AEROUT_POP_TSTEP   (CTRL_AEROUT_POP_TSTEP),
        .CTRL_POST_NEURON_ADDRESS(CTRL_POST_NEURON_ADDRESS),
        .CTRL_AEROUT_TREF_FINISH (CTRL_AEROUT_TREF_FINISH),
        // Inputs from neurons ------------------------------------
        .POST_NEUR_S_CNT         (POST_NEUR_S_CNT),
        // Output to controller ---------------------------
        .AEROUT_CTRL_FINISH      (AEROUT_CTRL_FINISH),
        // Output 8-bit AER link --------------------------
        .AEROUT_ADDR             (AEROUT_ADDR),
        .AEROUT_REQ              (AEROUT_REQ),
        .AEROUT_ACK              (AEROUT_ACK),
        .GOODNESS                (GOODNESS),
        .ONE_SAMPLE_FINISH       (ONE_SAMPLE_FINISH)
    );

    //----------------------------------------------------------------------------------
    //	Controller
    //----------------------------------------------------------------------------------


    controller #(
        .TIME_STEP                (TIME_STEP),
        .INPUT_NEURON             (INPUT_NEURON),
        .OUTPUT_NEURON            (OUTPUT_NEURON),
        .AER_IN_WIDTH                (AER_IN_WIDTH),
        .PRE_NEUR_ADDR_WIDTH      (PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH (PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH (PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH     (POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH(POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH(POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL       (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH      (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH     (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH      (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH(POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH     (SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH     (SYN_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH             (WEIGHT_WIDTH)
    ) controller_inst (
        // Global Inputs ------------------------------------------
        .CLK     (CLK),       // 时钟信号
        .RST     (RST_sync),  // 复位信号
        .IS_TRAIN(IS_TRAIN),  // 训练模式

        // Inputs from AER ----------------------------------------
        .AERIN_ADDR(AERIN_ADDR),  // AER 输入地址
        .AERIN_REQ (AERIN_REQ),   // AER 输入请求信号
        .AERIN_ACK (AERIN_ACK),   // AER 输入应答信号

        // Control Interface for Readback -------------------------
        .CTRL_READBACK_EVENT(CTRL_READBACK_EVENT),  // 读取回传事件
        .CTRL_PROG_EVENT    (CTRL_PROG_EVENT),      // 编程事件
        .CTRL_SPI_ADDR      (CTRL_SPI_ADDR),        // SPI 地址
        .CTRL_OP_CODE       (CTRL_OP_CODE),         // 操作码

        // Inputs from SPI Configuration Registers ----------------
        .SPI_GATE_ACTIVITY     (SPI_GATE_ACTIVITY),       // SPI 激活控制信号
        .SPI_GATE_ACTIVITY_sync(SPI_GATE_ACTIVITY_sync),  // SPI 激活同步信号
        .SPI_MAX_NEUR          (SPI_MAX_NEUR),            // 最大神经元数目

        // Inputs from Scheduler ----------------------------------
        .SCHED_EMPTY   (SCHED_EMPTY),    // 调度器是否为空
        .SCHED_FULL    (SCHED_FULL),     // 调度器是否满
        .SCHED_DATA_OUT(SCHED_DATA_OUT), // 调度器输出数据

        // Inputs from AER Output ---------------------------------
        .AEROUT_CTRL_BUSY  (AEROUT_CTRL_BUSY),   // AER 输出控制是否忙
        .AEROUT_CTRL_FINISH(AEROUT_CTRL_FINISH), // AER 输出控制是否完成

        // Outputs to Synaptic Core -------------------------------
        .CTRL_SYNARRAY_ADDR(CTRL_SYNARRAY_ADDR),  // 突触数组地址
        .CTRL_SYNARRAY_CS  (CTRL_SYNARRAY_CS),    // 突触数组选择信号
        .CTRL_SYNARRAY_WE  (CTRL_SYNARRAY_WE),    // 突触数组写使能信号

        .CTRL_GRAD_ARRAY_CS(CTRL_GRAD_ARRAY_CS),
        .CTRL_GRAD_ARRAY_WE(CTRL_GRAD_ARRAY_WE),

        .CTRL_SYNA_WR_EVENT (CTRL_SYNA_WR_EVENT),  // 突触写事件
        .CTRL_SYNA_RD_EVENT (CTRL_SYNA_RD_EVENT),  // 突触读事件
        .CTRL_SYNA_PROG_DATA(CTRL_SYNA_PROG_DATA), // 突触编程数据

        // Outputs to Neurons -------------------------------------
        .CTRL_WR_NEUR_EVENT      (CTRL_WR_NEUR_EVENT),        // 写神经元事件
        .CTRL_RD_NEUR_EVENT      (CTRL_RD_NEUR_EVENT),        // 读神经元事件
        .CTRL_POST_NEUR_PROG_DATA(CTRL_POST_NEUR_PROG_DATA),  // 神经元编程数据
        .CTRL_PRE_NEURON_ADDRESS (CTRL_PRE_NEURON_ADDRESS),   // 预神经元地址
        .CTRL_POST_NEURON_ADDRESS(CTRL_POST_NEURON_ADDRESS),  // 后神经元地址
        .CTRL_PRE_NEUR_CS        (CTRL_PRE_NEUR_CS),          // 预神经元选择信号
        .CTRL_PRE_NEUR_WE        (CTRL_PRE_NEUR_WE),          // 预神经元写使能信号
        .CTRL_POST_NEUR_CS       (CTRL_POST_NEUR_CS),         // 后神经元选择信号
        .CTRL_POST_NEUR_WE       (CTRL_POST_NEUR_WE),         // 后神经元写使能信号
        .CTRL_PRE_CNT_EN         (CTRL_PRE_CNT_EN),           // 预计数使能信号
        .CURRENT_TIME_STEP       (CURRENT_TIME_STEP),

        // Training and Inference Events --------------------------
        .CTRL_NEUR_EVENT (CTRL_NEUR_EVENT),   // 神经元事件
        .CTRL_TSTEP_EVENT(CTRL_TSTEP_EVENT),  // 时间步事件
        .CTRL_TREF_EVENT (CTRL_TREF_EVENT),   // 时间参考事件

        // Outputs to Scheduler -----------------------------------
        .CTRL_SCHED_POP_N   (CTRL_SCHED_POP_N),     // 调度器弹出神经元事件
        .CTRL_SCHED_ADDR    (CTRL_SCHED_ADDR),      // 调度器地址
        .CTRL_SCHED_EVENT_IN(CTRL_SCHED_EVENT_IN),  // 调度器输入事件
        .CTRL_SCHED_VIRTS   (CTRL_SCHED_VIRTS),     // 虚拟事件权重值

        // Outputs to AER Output -----------------------------------
        .CTRL_AEROUT_POP_NEUR   (CTRL_AEROUT_POP_NEUR),     // AER 输出弹出神经元事件
        .CTRL_AEROUT_PUSH_NEUR  (CTRL_AEROUT_PUSH_NEUR),    // AER 输出推送神经元事件
        .CTRL_AEROUT_POP_TSTEP  (CTRL_AEROUT_POP_TSTEP),    // AER 输出弹出时间步事件
        .CTRL_AEROUT_TREF_FINISH(CTRL_AEROUT_TREF_FINISH),  // AER 输出时间参考完成事件
        .ctrl_state             (ctrl_state),
        // Output to GOODNESS moving average module------------------
        .GOODNESS_ACC_VALID     (GOODNESS_ACC_VALID),
        .GOODNESS_CLEAR         (GOODNESS_CLEAR)
    );

    //----------------------------------------------------------------------------------
    //	Scheduler
    //----------------------------------------------------------------------------------
    // 实例化 scheduler 模块
    scheduler #(
        .TIME_STEP                (TIME_STEP),
        .INPUT_NEURON             (INPUT_NEURON),
        .OUTPUT_NEURON            (OUTPUT_NEURON),
        .AER_IN_WIDTH                (AER_IN_WIDTH),
        .PRE_NEUR_ADDR_WIDTH      (PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH (PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH (PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH     (POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH(POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH(POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL       (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH      (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH     (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH      (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH(POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH     (SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH     (SYN_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH             (WEIGHT_WIDTH)
    ) scheduler_inst (
        .CLK                (CLK),
        .RSTN               (RSTN_sync),
        .CTRL_SCHED_POP_N   (CTRL_SCHED_POP_N),
        .CTRL_SCHED_VIRTS   (CTRL_SCHED_VIRTS),
        .CTRL_SCHED_ADDR    (CTRL_SCHED_ADDR),
        .CTRL_SCHED_EVENT_IN(CTRL_SCHED_EVENT_IN),
        .SPI_OPEN_LOOP      (SPI_OPEN_LOOP),
        .SCHED_EMPTY        (SCHED_EMPTY),
        .SCHED_FULL         (SCHED_FULL),
        .SCHED_DATA_OUT     (SCHED_DATA_OUT)
    );

    //----------------------------------------------------------------------------------
    //	Synaptic core
    //----------------------------------------------------------------------------------
    synaptic_core #(
        .TIME_STEP                (TIME_STEP),
        .INPUT_NEURON             (INPUT_NEURON),
        .OUTPUT_NEURON            (OUTPUT_NEURON),
        .AER_IN_WIDTH                (AER_IN_WIDTH),
        .PRE_NEUR_ADDR_WIDTH      (PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH (PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH (PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH     (POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH(POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH(POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL       (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH      (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH     (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH      (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH(POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH     (SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH     (SYN_ARRAY_ADDR_WIDTH),
        .GRAD_ARRAY_DATA_WIDTH    (GRAD_ARRAY_DATA_WIDTH),
        .GRAD_ARRAY_ADDR_WIDTH    (GRAD_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH             (WEIGHT_WIDTH),
        .GRAD_WIDTH               (GRAD_WIDTH),
        .GOODNESS_WIDTH           (GOODNESS_WIDTH)
    ) u_synaptic_core (
        .IS_POS                  (IS_POS),
        .IS_TRAIN                (IS_TRAIN),        
        .AVG_GOODNESS            (AVG_GOODNESS),
        // Global inputs ------------------------------------------
        .CLK                     (CLK),
        // Inputs from controller ---------------------------------
        .CTRL_SYNARRAY_CS        (CTRL_SYNARRAY_CS),
        .CTRL_SYNARRAY_WE        (CTRL_SYNARRAY_WE),
        .CTRL_SYNARRAY_ADDR      (CTRL_SYNARRAY_ADDR),
        .CTRL_POST_NEURON_ADDRESS(CTRL_POST_NEURON_ADDRESS),

        .CTRL_GRAD_ARRAY_CS(CTRL_GRAD_ARRAY_CS),
        .CTRL_GRAD_ARRAY_WE(CTRL_GRAD_ARRAY_WE),

        .CTRL_NEUR_EVENT (CTRL_NEUR_EVENT),
        .CTRL_TSTEP_EVENT(CTRL_TSTEP_EVENT),
        .CTRL_TREF_EVENT (CTRL_TREF_EVENT),
        // Inputs from neurons ------------------------------------
        .PRE_NEUR_S_CNT  (PRE_NEUR_S_CNT),
        .POST_NEUR_S_CNT (POST_NEUR_S_CNT),
        // Outputs ------------------------------------------------
        .SYNARRAY_RDATA  (SYNARRAY_RDATA)
    );


    //----------------------------------------------------------------------------------
    //	Neuron core
    //----------------------------------------------------------------------------------
    neuron_core #(
        .TIME_STEP                (TIME_STEP),
        .INPUT_NEURON             (INPUT_NEURON),
        .OUTPUT_NEURON            (OUTPUT_NEURON),
        .AER_IN_WIDTH                (AER_IN_WIDTH),
        .PRE_NEUR_ADDR_WIDTH      (PRE_NEUR_ADDR_WIDTH),
        .PRE_NEUR_WORD_ADDR_WIDTH (PRE_NEUR_WORD_ADDR_WIDTH),
        .PRE_NEUR_BYTE_ADDR_WIDTH (PRE_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_ADDR_WIDTH     (POST_NEUR_ADDR_WIDTH),
        .POST_NEUR_WORD_ADDR_WIDTH(POST_NEUR_WORD_ADDR_WIDTH),
        .POST_NEUR_BYTE_ADDR_WIDTH(POST_NEUR_BYTE_ADDR_WIDTH),
        .POST_NEUR_PARALLEL       (POST_NEUR_PARALLEL),
        .PRE_NEUR_DATA_WIDTH      (PRE_NEUR_DATA_WIDTH),
        .POST_NEUR_DATA_WIDTH     (POST_NEUR_DATA_WIDTH),
        .POST_NEUR_MEM_WIDTH      (POST_NEUR_MEM_WIDTH),
        .POST_NEUR_SPIKE_CNT_WIDTH(POST_NEUR_SPIKE_CNT_WIDTH),
        .SYN_ARRAY_DATA_WIDTH     (SYN_ARRAY_DATA_WIDTH),
        .SYN_ARRAY_ADDR_WIDTH     (SYN_ARRAY_ADDR_WIDTH),
        .WEIGHT_WIDTH             (WEIGHT_WIDTH)
    ) neuron_core_inst (
        // Global inputs ------------------------------------------
        .CLK  (CLK),  // 时钟信号
        .RST_N(RST),  // 复位信号

        // Synaptic inputs ----------------------------------------
        .SYNARRAY_RDATA(SYNARRAY_RDATA),  // 从突触数组读取的输入数据

        // Controller inputs ----------------------------------------
        .CTRL_POST_NEUR_PROG_DATA(CTRL_POST_NEUR_PROG_DATA),  // 神经元编程数据
        .CTRL_PRE_NEURON_ADDRESS (CTRL_PRE_NEURON_ADDRESS),   // 预神经元地址
        .CTRL_POST_NEURON_ADDRESS(CTRL_POST_NEURON_ADDRESS),  // 后神经元地址
        .CTRL_WR_NEUR_EVENT      (CTRL_WR_NEUR_EVENT),        // 写神经元事件
        .CTRL_RD_NEUR_EVENT      (CTRL_RD_NEUR_EVENT),        // 读神经元事件
        .CTRL_NEUR_EVENT         (CTRL_NEUR_EVENT),           // 神经元事件
        .CTRL_TSTEP_EVENT        (CTRL_TSTEP_EVENT),          // 时间步事件
        .CTRL_TREF_EVENT         (CTRL_TREF_EVENT),           // 时间参考事件
        .CTRL_PRE_NEUR_CS        (CTRL_PRE_NEUR_CS),          // 预神经元选择信号
        .CTRL_PRE_NEUR_WE        (CTRL_PRE_NEUR_WE),          // 预神经元写使能信号
        .CTRL_POST_NEUR_CS       (CTRL_POST_NEUR_CS),         // 后神经元选择信号
        .CTRL_POST_NEUR_WE       (CTRL_POST_NEUR_WE),         // 后神经元写使能信号
        .CTRL_PRE_CNT_EN         (CTRL_PRE_CNT_EN),           // 预计数器使能信号
        .CURRENT_TIME_STEP       (CURRENT_TIME_STEP),

        // SPI inputs ----------------------------------------
        .SPI_GATE_ACTIVITY_sync(SPI_GATE_ACTIVITY_sync),  // SPI激活同步信号
        .SPI_POST_NEUR_ADDR    (SPI_POST_NEUR_ADDR),      // SPI后神经元地址

        // Outputs ----------------------------------------
        .NEUR_STATE     (NEUR_STATE),      // 神经元状态
        .NEUR_EVENT_OUT (NEUR_EVENT_OUT),  // 神经元事件输出
        .PRE_NEUR_S_CNT (PRE_NEUR_S_CNT),  // 预神经元脉冲计数
        .POST_NEUR_S_CNT(POST_NEUR_S_CNT),
        .POST_NEUR_MEM_BUS(POST_NEUR_MEM_BUS)
    );







endmodule


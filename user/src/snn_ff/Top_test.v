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
// Last modified Date:     2025/02/26 16:10:53
// Last Version:           V1.0
// Descriptions:           
//----------------------------------------------------------------------------------------
// Created by:             Sephiroth
// Created date:           2025/02/26 16:10:53
// mail      :             1245598043@qq.com
// Version:                V1.0
// TEXT NAME:              Top_test.v
// PATH:                   D:\MyProject\FPGA_prj\SNN_FFSTBP\rtl\snn_ff\Top_test.v
// Descriptions:           
//                         
//----------------------------------------------------------------------------------------
//****************************************************************************************//

module Top_test
#(
    parameter TIME_STEP = 8,
    parameter INPUT_NEURON = 784,
    parameter OUTPUT_NEURON = 256,
    parameter AER_WIDTH = 12,

    parameter POST_NEUR_PARALLEL = 8,

    parameter PRE_NEUR_ADDR_WIDTH = 10,
    parameter POST_NEUR_ADDR_WIDTH = 10,

    // parameter PRE_NEUR_DATA_WIDTH = 8, // 单个突触前神经元脉冲计数数据位宽
    parameter POST_NEUR_DATA_WIDTH = 20, // 单个突触后神经元状态数据位宽
    parameter POST_NEUR_MEM_WIDTH = 13, // 单个突触后神经元膜电位数据位宽
    // parameter POST_NEUR_SPIKE_CNT_WIDTH = 6, // 单个突触后神经元脉冲计数数据位宽
    parameter WEIGHT_WIDTH = 9, // 单个突触权重数据位宽
    parameter GRAD_WIDTH = 9
)
(
    input  wire                         CLK                        ,
    input  wire                         RST                        ,
    input  wire        [ 11: 0]         AERIN_ADDR                 ,
    input  wire                         AERIN_REQ                  ,
    input  wire                         IS_POS                     ,
    input  wire                         IS_TRAIN                   ,
    output wire                         AERIN_ACK                  ,
    output wire        [  31: 0]        GOODNESS                   ,
    output wire                         PROCESS_DONE
);
    
    parameter PRE_NEUR_WORD_ADDR_WIDTH= 10;
    parameter PRE_NEUR_BYTE_ADDR_WIDTH = 0;
    parameter POST_NEUR_BYTE_ADDR_WIDTH = $clog2(POST_NEUR_PARALLEL);
    parameter POST_NEUR_WORD_ADDR_WIDTH= POST_NEUR_ADDR_WIDTH - POST_NEUR_BYTE_ADDR_WIDTH;
    parameter SYN_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH; // 突触阵列数据位宽
    parameter SYN_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL); // 突触阵列地址位宽 
    parameter GRAD_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * GRAD_WIDTH; // 突触梯度阵列数据位宽
    parameter GRAD_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL); // 突触梯度阵列地址位宽
    parameter POST_NEUR_SPIKE_CNT_WIDTH = TIME_STEP; // 单个突触后神经元脉冲计数数据位宽
    parameter PRE_NEUR_DATA_WIDTH = TIME_STEP; // 单个突触前神经元脉冲计数数据位宽

    wire               [AER_WIDTH-1: 0]        AEROUT_ADDR                 ;
    wire                                AEROUT_REQ                  ;
    wire                                AEROUT_ACK                  ;
    reg                                 AEROUT_ACK_reg              ;
    reg                [   5: 0]        AEROUT_ACK_delay            ;
    
    wire                                ONE_SAMPLE_FINISH           ;
    wire                                SCHED_FULL                  ;
    
    assign PROCESS_DONE = ONE_SAMPLE_FINISH;
ODIN_ffstdp#(
    .TIME_STEP                             (TIME_STEP          ),
    .INPUT_NEURON                          (INPUT_NEURON       ),
    .OUTPUT_NEURON                         (OUTPUT_NEURON      ),
    .AER_WIDTH                             (AER_WIDTH          ),
    .PRE_NEUR_ADDR_WIDTH                   (PRE_NEUR_ADDR_WIDTH),
    .PRE_NEUR_WORD_ADDR_WIDTH              (PRE_NEUR_WORD_ADDR_WIDTH),
    .PRE_NEUR_BYTE_ADDR_WIDTH              (PRE_NEUR_BYTE_ADDR_WIDTH),
    .POST_NEUR_ADDR_WIDTH                  (POST_NEUR_ADDR_WIDTH),
    .POST_NEUR_WORD_ADDR_WIDTH             (POST_NEUR_WORD_ADDR_WIDTH),
    .POST_NEUR_BYTE_ADDR_WIDTH             (POST_NEUR_BYTE_ADDR_WIDTH),
    .POST_NEUR_PARALLEL                    (POST_NEUR_PARALLEL ),
    .PRE_NEUR_DATA_WIDTH                   (PRE_NEUR_DATA_WIDTH),
    .POST_NEUR_DATA_WIDTH                  (POST_NEUR_DATA_WIDTH),
    .POST_NEUR_MEM_WIDTH                   (POST_NEUR_MEM_WIDTH),
    .POST_NEUR_SPIKE_CNT_WIDTH             (POST_NEUR_SPIKE_CNT_WIDTH),
    .SYN_ARRAY_DATA_WIDTH     (SYN_ARRAY_DATA_WIDTH),
    .SYN_ARRAY_ADDR_WIDTH     (SYN_ARRAY_ADDR_WIDTH),
    .GRAD_ARRAY_DATA_WIDTH    (GRAD_ARRAY_DATA_WIDTH),
    .GRAD_ARRAY_ADDR_WIDTH    (GRAD_ARRAY_ADDR_WIDTH),
    .WEIGHT_WIDTH             (WEIGHT_WIDTH),
    .GRAD_WIDTH               (GRAD_WIDTH)
)
 u_ODIN_ffstdp(
// Global input     -------------------------------
    .CLK                                (CLK                       ),
    .RST                                (RST                       ),
    .IS_POS                             (IS_POS                    ),// 0: negative, 1: positive
    .IS_TRAIN                           (IS_TRAIN                  ),// 0: inference, 1: training
    .AVG_GOODNESS                       (AVG_GOODNESS              ),
// Input 12-bit AER -------------------------------
    .AERIN_ADDR                         (AERIN_ADDR                ),
    .AERIN_REQ                          (AERIN_REQ                 ),
    .AERIN_ACK                          (AERIN_ACK                 ),
// Output 10-bit AER -------------------------------
    .AEROUT_ADDR                        (AEROUT_ADDR               ),
    .AEROUT_REQ                         (AEROUT_REQ                ),
    .AEROUT_ACK                         (AEROUT_ACK                ),
    .GOODNESS                           (GOODNESS                  ),
    .ONE_SAMPLE_FINISH                  (ONE_SAMPLE_FINISH         )
);

always @(posedge CLK or posedge RST)
    begin
        if(RST)
            AEROUT_ACK_reg <= 1'b0;
        else if(AEROUT_REQ)
            AEROUT_ACK_reg <= 1'b1;
        else
            AEROUT_ACK_reg <= 1'b0;
    end
always @(posedge CLK or posedge RST)
    begin
        if(RST)
            AEROUT_ACK_delay <= 6'b0;
        else
            AEROUT_ACK_delay <= {AEROUT_ACK_delay[4:0],AEROUT_ACK_reg};
    end
assign AEROUT_ACK = AEROUT_ACK_delay[5];                                                           
endmodule
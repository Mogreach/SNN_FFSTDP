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

module Top_test(
    input  wire                         CLK                        ,
    input  wire                         RST                        ,
    input  wire        [ 11: 0]         AERIN_ADDR                 ,
    input  wire                         AERIN_REQ                  ,
    input  wire                         IS_POS                     ,
    input  wire                         IS_TRAIN                   ,
    output wire                         AERIN_ACK                  ,
    output wire        [  31: 0]        GOODNESS                   ,
    output wire                         PROCESS_DONE,
    output wire       [  3:0]                  ctrl_state
);
    parameter                           M                           = 12                   ;
    parameter                           N                           = 784                  ;

    reg                                 SCK                         ;
    reg                                 MOSI                        ;
    wire                                MISO                        ;

    wire               [ M-1: 0]        AEROUT_ADDR                 ;
    wire                                AEROUT_REQ                  ;
    wire                                AEROUT_ACK                  ;
    reg                                 AEROUT_ACK_reg              ;
    reg                [   5: 0]        AEROUT_ACK_delay            ;
    
    wire                                ONE_SAMPLE_FINISH           ;
    wire                                SCHED_FULL                  ;
    
    assign PROCESS_DONE = ONE_SAMPLE_FINISH;
ODIN_ffstdp#(
   .N              (784            ),
   .M              (10             )
)
 u_ODIN_ffstdp(
// Global input     -------------------------------
    .CLK                                (CLK                       ),
    .RST                                (RST                       ),
    .IS_POS                             (IS_POS                    ),// 0: negative, 1: positive
    .IS_TRAIN                           (IS_TRAIN                  ),// 0: inference, 1: training
// SPI slave        -------------------------------
    .SCK                                (SCK                       ),
    .MOSI                               (MOSI                      ),
    .MISO                               (MISO                      ),
// Input 12-bit AER -------------------------------
    .AERIN_ADDR                         (AERIN_ADDR                ),
    .AERIN_REQ                          (AERIN_REQ                 ),
    .AERIN_ACK                          (AERIN_ACK                 ),
// Output 10-bit AER -------------------------------
    .AEROUT_ADDR                        (AEROUT_ADDR               ),
    .AEROUT_REQ                         (AEROUT_REQ                ),
    .AEROUT_ACK                         (AEROUT_ACK                ),
    .GOODNESS                           (GOODNESS                  ),
    .ONE_SAMPLE_FINISH                  (ONE_SAMPLE_FINISH         ),
// Debug ------------------------------------------
    .SCHED_FULL                         (SCHED_FULL                ),
    .ctrl_state                         (ctrl_state                )
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
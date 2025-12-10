// Copyright (C) 2019-2022, Université catholique de Louvain (UCLouvain, Belgium), University of Zürich (UZH, Switzerland),
//         Katholieke Universiteit Leuven (KU Leuven, Belgium), and Delft University of Technology (TU Delft, Netherlands).
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// Licensed under the Solderpad Hardware License v 2.1 (the “License”); you may not use this file except in compliance
// with the License, or, at your option, the Apache License version 2.0. You may obtain a copy of the License at
// https://solderpad.org/licenses/SHL-2.1/
//
// Unless required by applicable law or agreed to in writing, any work distributed under the License is distributed on
// an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
//------------------------------------------------------------------------------
//
// "aer_out.v" - Output AER module, custom monitoring mode from ODIN was removed
// 
// Project: tinyODIN - A low-cost digital spiking neuromorphic processor adapted from ODIN.
//
// Author:  C. Frenkel, Delft University of Technology
//
// Cite/paper: C. Frenkel, M. Lefebvre, J.-D. Legat and D. Bol, "A 0.086-mm² 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning
//             Digital Spiking Neuromorphic Processor in 28-nm CMOS," IEEE Transactions on Biomedical Circuits and Systems,
//             vol. 13, no. 1, pp. 145-158, 2019.
//
//------------------------------------------------------------------------------


module aer_out #(
	parameter N = 256,
	parameter M = 10
)(

    // Global input ----------------------------------- 
    input  wire           CLK,
    input  wire           RST,
    // Inputs from SPI configuration latches ----------
    input  wire           SPI_GATE_ACTIVITY_sync,
    input  wire           SPI_AER_SRC_CTRL_nNEUR,
    // Neuron data inputs -----------------------------
    input  wire [3:0]     NEUR_EVENT_OUT,
    // Input from scheduler ---------------------------
    input  wire [   11:0] SCHED_DATA_OUT,
    // Input from controller --------------------------
    input  wire           CTRL_TREF_EVENT,
    input  wire           CTRL_POST_NEUR_CS, 
    input  wire           CTRL_POST_NEUR_WE, 
    input  wire           CTRL_AEROUT_PUSH_NEUR,
    input  wire           CTRL_AEROUT_POP_NEUR,
    input  wire           CTRL_AEROUT_POP_TSTEP,
    input  wire [9:0]     CTRL_POST_NEURON_ADDRESS,
    input  wire           CTRL_AEROUT_TREF_FINISH,
    // Inputs from neurons ------------------------------------
    input wire [6:0]      POST_NEUR_S_CNT_0,
    input wire [6:0]      POST_NEUR_S_CNT_1,
    input wire [6:0]      POST_NEUR_S_CNT_2,
    input wire [6:0]      POST_NEUR_S_CNT_3,
    
    // Output to controller ---------------------------
    output wire           AEROUT_CTRL_FINISH,
    
	// Output 8-bit AER link --------------------------
	output reg  [  M-1:0] AEROUT_ADDR, 
	output reg  	      AEROUT_REQ,
	input  wire 	      AEROUT_ACK,

    output reg [31:0]     GOODNESS,
    output wire           ONE_SAMPLE_FINISH
);

    reg                [  31: 0]        goodness                    ;
    reg                                 AEROUT_ACK_sync_int,      AEROUT_ACK_sync,AEROUT_ACK_sync_del;
    reg                                 aer_out_addr_last,        aer_out_addr_last_int;
    reg                                 aer_out_trans               ;//AEROUT输出事件中
    reg                                 fifo_rd_en_int              ;
    reg                                 goodness_en_d0              ;
    reg                                 goodness_en_d1              ;
    reg                [   5: 0]        ctrl_tref_finish_delay      ;
    // 路径太长，违反时序，增加一级寄存器
    reg                                 fifo_wr_en_d0               ;
    reg                [  47: 0]        aer_out_fifo_din_d0         ;

    wire                                aer_out_start               ;
    wire                                AEROUT_ACK_sync_negedge     ;
    wire                                aer_out_addr_last_negedge   ;
    wire                                rst_activity                ;
    wire                                fifo_rd_en                  ;
    wire                                fifo_wr_en                  ;
    wire                                fifo_empty                  ;
    wire                                fifo_full                   ;
    wire                                goodness_en                 ;
    wire                                ctrl_tref_finish_delay_posedge  ;
    
    
    
    wire               [  47: 0]        aer_out_fifo_din            ;
    wire               [  11: 0]        aer_out_fifo_dout           ;
    wire               [   6: 0]        post_neur_cnt[3:0]          ;
    wire               [   9: 0]        post_neur_goodness[3:0]     ;
    wire               [  10: 0]        post_neur_goodness_add1     ;
    wire               [  10: 0]        post_neur_goodness_add2     ;
    wire               [  11: 0]        post_neur_goodness_sum      ;

    assign                              rst_activity                = RST;
    // AER空闲，fifo不空，且此时fifo_out不是无效事件
    assign                              fifo_rd_en                  = CTRL_AEROUT_POP_NEUR & !AEROUT_REQ & !AEROUT_ACK_sync & !fifo_empty & !aer_out_start;
    assign                              fifo_wr_en                  = CTRL_AEROUT_PUSH_NEUR & !fifo_full;

    assign                              AEROUT_ACK_sync_negedge     = !AEROUT_ACK_sync & AEROUT_ACK_sync_del;
    assign                              aer_out_addr_last_negedge   = !aer_out_addr_last & aer_out_addr_last_int;

    assign                              aer_out_start               = fifo_rd_en_int && (!(&aer_out_fifo_dout[11:10]));// 无效事件11则不传输
    assign                              AEROUT_CTRL_FINISH          = aer_out_addr_last_negedge;

    assign                              post_neur_cnt[0]            = POST_NEUR_S_CNT_0;
    assign                              post_neur_cnt[1]            = POST_NEUR_S_CNT_1;
    assign                              post_neur_cnt[2]            = POST_NEUR_S_CNT_2;
    assign                              post_neur_cnt[3]            = POST_NEUR_S_CNT_3;
    assign                              ctrl_tref_finish_delay_posedge= !ctrl_tref_finish_delay[5] && ctrl_tref_finish_delay[4];
    assign                              ONE_SAMPLE_FINISH           = ctrl_tref_finish_delay_posedge;
    assign                              goodness_en                 = CTRL_POST_NEUR_WE && CTRL_TREF_EVENT;

    genvar i;
    generate
        for (i = 0; i<4; i=i+1) begin
            assign aer_out_fifo_din[12*(4-i)-1:12*(3-i)]= NEUR_EVENT_OUT[i]? {2'b00,(CTRL_POST_NEURON_ADDRESS+i)} : {2'b11,10'd0};
            goodness_mult goodness_square (
            .CLK                                (CLK                       ),// input wire CLK
            .A                                  (post_neur_cnt[i]          ),// input wire [4 : 0] A
            .B                                  (post_neur_cnt[i]          ),// input wire [4 : 0] B
            .P                                  (post_neur_goodness[i]     ) // output wire [9 : 0] P
                    );
        end
    endgenerate

    goodness_adder_1 goodness_adder1 (
    .A                                  (post_neur_goodness[0]     ),// input wire [9 : 0] A
    .B                                  (post_neur_goodness[1]     ),// input wire [9 : 0] B
    .S                                  (post_neur_goodness_add1   ) // output wire [10 : 0] S
    );
    goodness_adder_1 goodness_adder2 (
    .A                                  (post_neur_goodness[2]     ),// input wire [9 : 0] A
    .B                                  (post_neur_goodness[3]     ),// input wire [9 : 0] B
    .S                                  (post_neur_goodness_add2   ) // output wire [10 : 0] S
    );
    goodness_adder_2 goodness_adder3 (
    .A                                  (post_neur_goodness_add1   ),// input wire [10 : 0] A
    .B                                  (post_neur_goodness_add2   ),// input wire [10 : 0] B
    .S                                  (post_neur_goodness_sum    ) // output wire [11 : 0] S
    );


    always @(posedge CLK or posedge rst_activity)           
        begin                                        
            if(rst_activity || ctrl_tref_finish_delay_posedge)                               
                GOODNESS <= 'd0;                         
            else if(goodness_en_d1)                                
                GOODNESS <= GOODNESS + post_neur_goodness_sum;              
            else   
                GOODNESS <= GOODNESS;                                  
        end                                          


    // Sync barrier
    always @(posedge CLK, posedge rst_activity) begin
        if (rst_activity) begin
            AEROUT_ACK_sync_int <= 1'b0;
            AEROUT_ACK_sync	    <= 1'b0;
            AEROUT_ACK_sync_del <= 1'b0;
            aer_out_addr_last_int <= 1'b0;
            fifo_rd_en_int <= 1'b0;
            goodness_en_d0 <= 1'b0;
            goodness_en_d1 <= 1'b0;
            ctrl_tref_finish_delay <= 6'b0;
            // aer_out_fifo_din_d0 <= 48'b0;
            // fifo_wr_en_d0 <= 1'b0;
        end
        else begin
            AEROUT_ACK_sync_int <= AEROUT_ACK;
            AEROUT_ACK_sync	    <= AEROUT_ACK_sync_int;
            AEROUT_ACK_sync_del <= AEROUT_ACK_sync;
            aer_out_addr_last_int <= aer_out_addr_last;
            fifo_rd_en_int <= fifo_rd_en;
            goodness_en_d0 <= goodness_en;
            goodness_en_d1 <= goodness_en_d0;
            ctrl_tref_finish_delay <= {ctrl_tref_finish_delay[4:0],CTRL_AEROUT_TREF_FINISH};
            // aer_out_fifo_din_d0 <= aer_out_fifo_din;
            // fifo_wr_en_d0 <= fifo_wr_en;
        end
    end

                                                                            
    // Output AER interface
    always @(posedge CLK, posedge rst_activity) begin
        if (rst_activity) begin
            AEROUT_ADDR             <= 10'b0;
            AEROUT_REQ              <= 1'b0;
            aer_out_trans           <= 1'b0;
            aer_out_addr_last       <= 1'b0;
        end else begin
            // 只有当处于POP_TSTEP，且fifo已排空，且aerout不在传输中，且还没发送tstep_event，才开始发送tstep_event
            if (CTRL_AEROUT_POP_TSTEP && fifo_empty && !aer_out_trans && !AEROUT_CTRL_FINISH)begin
                AEROUT_ADDR      <= {2'b01,10'd0};
                AEROUT_REQ       <= 1'b1;
                aer_out_trans    <= 1'b1;
                aer_out_addr_last <= 1'b1;
            end else if ((aer_out_start) && !AEROUT_ACK_sync) begin
                AEROUT_ADDR      <= aer_out_fifo_dout;
                AEROUT_REQ       <= 1'b1;
                aer_out_trans <= 1'b1;
                aer_out_addr_last <= 1'b0;
            end else if (AEROUT_ACK_sync) begin
                AEROUT_REQ       <= 1'b0;
                aer_out_trans <= 1'b1;
                aer_out_addr_last <= aer_out_addr_last;
            end else if (AEROUT_ACK_sync_negedge) begin
                AEROUT_REQ       <= 1'b0;
                aer_out_trans <= 1'b0;
                aer_out_addr_last <= 1'b0;
            end
        end
    end
    
    aer_out_fifo aer_out_fifo_0 (
    .clk(CLK),      // input wire clk
    .srst(rst_activity),    // input wire srst
    .din(aer_out_fifo_din),      // input wire [47 : 0] din
    .wr_en(fifo_wr_en),  // input wire wr_en
    .rd_en(fifo_rd_en),  // input wire rd_en
    .dout(aer_out_fifo_dout),    // output wire [11 : 0] dout
    .full(fifo_full),    // output wire full
    .empty(fifo_empty)  // output wire empty
    );
    

endmodule 

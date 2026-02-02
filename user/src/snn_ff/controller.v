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
// "controller.v" - Controller module
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


module controller #(
    parameter TIME_STEP = 8,
    parameter INPUT_NEURON = 784,
    parameter OUTPUT_NEURON = 256,
    parameter AER_IN_WIDTH = 12,

    parameter PRE_NEUR_ADDR_WIDTH = 10,
    parameter PRE_NEUR_WORD_ADDR_WIDTH= 10,
    parameter PRE_NEUR_BYTE_ADDR_WIDTH = 0,

    parameter POST_NEUR_ADDR_WIDTH = 10,
    parameter POST_NEUR_WORD_ADDR_WIDTH= 8,
    parameter POST_NEUR_BYTE_ADDR_WIDTH = 2,
    parameter POST_NEUR_PARALLEL = 4,
    
    parameter PRE_NEUR_DATA_WIDTH = 8,
    parameter POST_NEUR_DATA_WIDTH = 32,
    parameter POST_NEUR_MEM_WIDTH = 12,
    parameter POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter SYN_ARRAY_DATA_WIDTH = 32,
    parameter SYN_ARRAY_ADDR_WIDTH = 16,
    parameter WEIGHT_WIDTH = 8
)(    

    // Global inputs ------------------------------------------
    input  wire           CLK,
    input  wire           RST,
    input  wire           IS_TRAIN,
    
    // Inputs from AER ----------------------------------------
    input  wire   [AER_IN_WIDTH-1:0] AERIN_ADDR,
    input  wire           AERIN_REQ,
    output reg            AERIN_ACK,
    
    // Control interface for readback -------------------------
    input  wire           CTRL_READBACK_EVENT,
    input  wire           CTRL_PROG_EVENT,
    input  wire [2*PRE_NEUR_ADDR_WIDTH-1:0] CTRL_SPI_ADDR,
    input  wire     [1:0] CTRL_OP_CODE,
    
    // Inputs from SPI configuration registers ----------------
    input  wire           SPI_GATE_ACTIVITY, 
    output reg            SPI_GATE_ACTIVITY_sync,
    input  wire   [PRE_NEUR_ADDR_WIDTH-1:0] SPI_MAX_NEUR,
    
    // Inputs from scheduler ----------------------------------
    input  wire           SCHED_EMPTY,
    input  wire           SCHED_FULL,
    input  wire    [AER_IN_WIDTH-1:0] SCHED_DATA_OUT,
    
    // Input from AER output ----------------------------------
    input  wire           AEROUT_CTRL_BUSY,
    input wire           AEROUT_CTRL_FINISH,
    
    // Outputs to synaptic core -------------------------------
    output reg    [ SYN_ARRAY_ADDR_WIDTH-1:0] CTRL_SYNARRAY_ADDR, 
    output reg            CTRL_SYNARRAY_CS,
    output reg            CTRL_SYNARRAY_WE,

    output reg CTRL_SYNA_WR_EVENT,
    output reg CTRL_SYNA_RD_EVENT,
    output reg[WEIGHT_WIDTH-1:0] CTRL_SYNA_PROG_DATA,
    
    // Outputs to neurons -------------------------------------
    //SPI控制读写事件
    output reg            CTRL_WR_NEUR_EVENT,
    output reg            CTRL_RD_NEUR_EVENT,
    // SPI控制编入数据
    output reg [SYN_ARRAY_DATA_WIDTH-1:0]     CTRL_POST_NEUR_PROG_DATA,
    //控制器神经元地址
    output reg  [PRE_NEUR_ADDR_WIDTH-1:0]      CTRL_PRE_NEURON_ADDRESS,
    output reg  [POST_NEUR_ADDR_WIDTH-1:0]      CTRL_POST_NEURON_ADDRESS,//突触后神经元地址
    output reg   CTRL_PRE_NEUR_CS,
    output reg   CTRL_PRE_NEUR_WE,
    output reg   CTRL_POST_NEUR_CS,
    output reg   CTRL_POST_NEUR_WE,
    output reg   CTRL_PRE_CNT_EN,
    output wire  CTRL_GRAD_ARRAY_CS,
    output wire  CTRL_GRAD_ARRAY_WE,

    //训练推理事件
    output reg  CTRL_NEUR_EVENT,
    output reg  CTRL_TSTEP_EVENT,
    output reg  CTRL_TREF_EVENT,

    // Outputs to pre-neuron
    output wire [$clog2(TIME_STEP)-1:0] CURRENT_TIME_STEP,
        
    // Outputs to scheduler -----------------------------------
    output reg            CTRL_SCHED_POP_N,
    output reg    [PRE_NEUR_ADDR_WIDTH-1:0] CTRL_SCHED_ADDR, // 神经元地址
    output reg            CTRL_SCHED_EVENT_IN, // 传入调度器的标志（PUSH状态）1表示外部事件，0表示传入内部事件（即输出突触后激活的事件） 
    output reg    [  1:0] CTRL_SCHED_VIRTS, // 虚拟事件的权重值
    
    // Output to AER output -----------------------------------
    output  reg            CTRL_AEROUT_POP_NEUR,
    output  reg            CTRL_AEROUT_PUSH_NEUR,
    output  reg            CTRL_AEROUT_POP_TSTEP,
    output  wire           CTRL_AEROUT_TREF_FINISH,
    // Output to GOODNESS moving average module------------------
    output  wire            GOODNESS_ACC_VALID,
    output  wire            GOODNESS_CLEAR,
    //debug
    output wire    [  3:0]                     ctrl_state                  
);

	//----------------------------------------------------------------------------------
	//	PARAMETERS 
	//----------------------------------------------------------------------------------
    localparam R_W_SRAM_CLK_neur_act = 3;
    localparam R_W_SRAM_CLK_neur_act_LOG2 = $clog2(R_W_SRAM_CLK_neur_act);
    localparam AER_TOTAL_CLK_neur_act = R_W_SRAM_CLK_neur_act * (OUTPUT_NEURON / POST_NEUR_PARALLEL) - 1;

    localparam R_W_SRAM_CLK_tstep_act = 2;
    localparam R_W_SRAM_CLK_tstep_act_LOG2 = $clog2(R_W_SRAM_CLK_tstep_act);
    localparam AER_TOTAL_CLK_tstep_act = R_W_SRAM_CLK_tstep_act * (OUTPUT_NEURON / POST_NEUR_PARALLEL) - 1;


    localparam R_W_SRAM_CLK_tref = 4;
    localparam R_W_SRAM_CLK_tref_LOG2 = $clog2(R_W_SRAM_CLK_tref);
    localparam AER_TOTAL_CLK_tref = R_W_SRAM_CLK_tref * (OUTPUT_NEURON / POST_NEUR_PARALLEL) - 1;
	// FSM states 
	localparam WAIT       = 4'd0; 
    localparam W_NEUR     = 4'd1;
    localparam R_NEUR     = 4'd2;
    localparam W_SYN      = 4'd3;
    localparam R_SYN      = 4'd4;
    localparam PUSH       = 4'd5;
    localparam POP_NEUR   = 4'd6;
    localparam NEUR_ACT   = 4'd7;
    localparam POP_NEUR_OUT  = 4'd8;
    localparam TSTEP_ACT  = 4'd9;
	localparam POP_TSTEP= 4'd10;
    localparam TREF       = 4'd11;
    localparam WAIT_SPIDN = 4'd12;
    localparam WAIT_REQDN = 4'd13;

	//----------------------------------------------------------------------------------
	//	REGS & WIRES
	//----------------------------------------------------------------------------------
    
    reg          AERIN_REQ_sync_int, AERIN_REQ_sync;
    reg          SPI_GATE_ACTIVITY_sync_int;
    reg          CTRL_READBACK_EVENT_sync_int, CTRL_READBACK_EVENT_sync;
    reg          CTRL_PROG_EVENT_sync_int, CTRL_PROG_EVENT_sync;

    wire         neuron_event,tstep_event, tref_event;
    wire         CTRL_TSTEP_EVENT_posedge;
    wire         tref_finish;
    
    reg  [ 31:0] ctrl_cnt;
    reg [R_W_SRAM_CLK_neur_act_LOG2-1:0]ctrl_neur_act_cnt;
    reg [R_W_SRAM_CLK_tstep_act_LOG2-1:0]ctrl_tstep_act_cnt;
    reg [R_W_SRAM_CLK_tref_LOG2-1:0]ctrl_tref_cnt;
    
    reg  [$clog2(TIME_STEP):0]  T_step_cnt;
    reg          CTRL_TSTEP_EVENT_int;
    reg  [$clog2(OUTPUT_NEURON)-1:0]  post_neur_cnt;
    reg          post_neur_cnt_inc;
    reg  [$clog2(INPUT_NEURON)-1:0]   pre_neur_cnt;
    reg          pre_neur_cnt_inc;
    reg  [  3:0] state, nextstate;

    
	//----------------------------------------------------------------------------------
	//	EVENT TYPE DECODING 
	//----------------------------------------------------------------------------------
    assign neuron_event   = !AERIN_ADDR[AER_IN_WIDTH-1] && !AERIN_ADDR[AER_IN_WIDTH-2];
    assign tstep_event    = !AERIN_ADDR[AER_IN_WIDTH-1] && AERIN_ADDR[AER_IN_WIDTH-2];
    assign tref_event     = AERIN_ADDR[AER_IN_WIDTH-1] && !AERIN_ADDR[AER_IN_WIDTH-2];
    assign CTRL_TSTEP_EVENT_negedge = !CTRL_TSTEP_EVENT & CTRL_TSTEP_EVENT_int;
    assign tref_finish = IS_TRAIN? (CTRL_TREF_EVENT && (pre_neur_cnt == INPUT_NEURON))
                                 : (CTRL_TREF_EVENT && (post_neur_cnt == (OUTPUT_NEURON / POST_NEUR_PARALLEL))); // 在推理更新状态，当突触前神经元计数到784时拉高，跳转至wait
	assign CTRL_AEROUT_TREF_FINISH = tref_finish;
    assign ctrl_state = state;

    assign CURRENT_TIME_STEP = T_step_cnt[$clog2(TIME_STEP)-1:0];
    assign GOODNESS_ACC_VALID = CTRL_TSTEP_EVENT && post_neur_cnt_inc;
    assign GOODNESS_CLEAR = tref_finish;

    assign CTRL_GRAD_ARRAY_CS = CTRL_SYNARRAY_CS;
    assign CTRL_GRAD_ARRAY_WE = CTRL_SYNARRAY_WE;
    //----------------------------------------------------------------------------------
	//	SYNC BARRIERS FROM AER AND FROM SPI
	//----------------------------------------------------------------------------------
    
   always @(posedge CLK, posedge RST) begin
		if(RST) begin
			AERIN_REQ_sync_int           <= 1'b0;
			AERIN_REQ_sync	             <= 1'b0;
            SPI_GATE_ACTIVITY_sync_int   <= 1'b0;
            SPI_GATE_ACTIVITY_sync       <= 1'b0;
            CTRL_READBACK_EVENT_sync_int <= 1'b0;
            CTRL_READBACK_EVENT_sync     <= 1'b0;
            CTRL_PROG_EVENT_sync_int     <= 1'b0;
            CTRL_PROG_EVENT_sync         <= 1'b0;
            CTRL_TSTEP_EVENT_int         <= 1'b0;
		end
		else begin
			AERIN_REQ_sync_int           <= AERIN_REQ;
			AERIN_REQ_sync	             <= AERIN_REQ_sync_int;
            SPI_GATE_ACTIVITY_sync_int   <= SPI_GATE_ACTIVITY;
            SPI_GATE_ACTIVITY_sync       <= SPI_GATE_ACTIVITY_sync_int;
            CTRL_READBACK_EVENT_sync_int <= CTRL_READBACK_EVENT;
            CTRL_READBACK_EVENT_sync     <= CTRL_READBACK_EVENT_sync_int;
            CTRL_PROG_EVENT_sync_int     <= CTRL_PROG_EVENT;
            CTRL_PROG_EVENT_sync         <= CTRL_PROG_EVENT_sync_int;
            CTRL_TSTEP_EVENT_int         <= CTRL_TSTEP_EVENT;
		end
	end
    
	//----------------------------------------------------------------------------------
	//	CONTROL FSM
	//----------------------------------------------------------------------------------
    
    // State register
	always @(posedge CLK, posedge RST)
	begin
		if   (RST) state <= WAIT;
		else       state <= nextstate;
	end
    //virt_event 与 tstep_event相似
    
	// Next state logic
	always @(*)
		case(state)
			WAIT 		:	
                            if (SCHED_FULL)                                                                 
                                //SCHED_DATA_OUT[11:10] == 2'b01
                                if( SCHED_DATA_OUT[AER_IN_WIDTH-1:AER_IN_WIDTH-2] == 2'b01)                nextstate = TSTEP_ACT;
                                else                                                                        nextstate = NEUR_ACT;
                            else if (AERIN_REQ_sync)
                                if (neuron_event || tstep_event)                                            nextstate = PUSH;
                                else                                                                        nextstate = WAIT;
                            else if (~SCHED_EMPTY)                                                          
                                if( SCHED_DATA_OUT[AER_IN_WIDTH-1:AER_IN_WIDTH-2] == 2'b01)                 nextstate = TSTEP_ACT;
                                else if (SCHED_DATA_OUT[AER_IN_WIDTH-1:AER_IN_WIDTH-2] == 2'b00)            nextstate = NEUR_ACT;
                                else                                                                        nextstate = WAIT;
                            else                                                                            nextstate = WAIT;
			TREF    	:   if      (tref_finish)                                                               nextstate = WAIT;
							else					                                                            nextstate = TREF;
            PUSH        :                                                                                       nextstate = WAIT_REQDN;
			//确保CTRL_SCHED_POP_N拉低一周期
                                                 //{SPI_MAX_NEUR,1'b1}
            NEUR_ACT    :   if      ((post_neur_cnt == OUTPUT_NEURON - POST_NEUR_PARALLEL) && (ctrl_neur_act_cnt == R_W_SRAM_CLK_neur_act - 1'b1))  nextstate = POP_NEUR;
							else					                                                            nextstate = NEUR_ACT;
            POP_NEUR    :   if      (~CTRL_SCHED_POP_N)                                                         nextstate = WAIT;
							else					                                                            nextstate = POP_NEUR;                
			TSTEP_ACT   :   if      ((post_neur_cnt == OUTPUT_NEURON - POST_NEUR_PARALLEL) && (ctrl_tstep_act_cnt == R_W_SRAM_CLK_tstep_act - 1'b1))nextstate = POP_NEUR_OUT;
							else					                                                            nextstate = TSTEP_ACT;
            POP_NEUR_OUT:   if      (~CTRL_SCHED_POP_N)                                                         nextstate =POP_TSTEP;
							else					                                                            nextstate = POP_NEUR_OUT;
            POP_TSTEP :     if      (AEROUT_CTRL_FINISH)                                   
                                if  ((T_step_cnt == TIME_STEP))                                                 nextstate = TREF;
                                else                                                                            nextstate = WAIT;
                            else                                                                                nextstate =POP_TSTEP;   
			WAIT_REQDN 	:   if      (~AERIN_REQ_sync)                                                           nextstate = WAIT;
							else					                                                            nextstate = WAIT_REQDN;
			default		:							                                                            nextstate = WAIT;
		endcase 
        
    // Control counter
	always @(posedge CLK, posedge RST)
		if      (RST)               ctrl_cnt <= 32'd0;
        else if ((state == WAIT) || (state == POP_TSTEP) || (CTRL_NEUR_EVENT && ctrl_cnt[$clog2(AER_TOTAL_CLK_neur_act+1)-1:0] == AER_TOTAL_CLK_neur_act))    
                                    ctrl_cnt <= 32'd0;
		else if (CTRL_NEUR_EVENT | CTRL_TSTEP_EVENT | CTRL_TREF_EVENT)
                                    ctrl_cnt <= ctrl_cnt + 32'd1;
        else                        ctrl_cnt <= ctrl_cnt;

    always @(posedge CLK, posedge RST) begin           
		if      (RST)               ctrl_neur_act_cnt <= 'd0;
        else if ((state == WAIT) || (CTRL_NEUR_EVENT && ctrl_neur_act_cnt == R_W_SRAM_CLK_neur_act - 1'b1))    
                                    ctrl_neur_act_cnt <= 'd0;
		else if (CTRL_NEUR_EVENT)
                                    ctrl_neur_act_cnt <= ctrl_neur_act_cnt + 1'b1;
        else                        ctrl_neur_act_cnt <= ctrl_neur_act_cnt;                                       
    end

    always @(posedge CLK, posedge RST) begin           
		if      (RST)               ctrl_tstep_act_cnt <= 'd0;
        else if ((state == WAIT) || (state == POP_TSTEP) || (CTRL_TSTEP_EVENT && ctrl_tstep_act_cnt == R_W_SRAM_CLK_tstep_act - 1'b1))    
                                    ctrl_tstep_act_cnt <= 'd0;
		else if (CTRL_TSTEP_EVENT)
                                    ctrl_tstep_act_cnt <= ctrl_tstep_act_cnt + 1'b1;
        else                        ctrl_tstep_act_cnt <= ctrl_tstep_act_cnt;                                       
    end

    always @(posedge CLK, posedge RST) begin           
		if      (RST)               ctrl_tref_cnt <= 'd0;
        else if ((state == WAIT) || (CTRL_TREF_EVENT && ctrl_tref_cnt == R_W_SRAM_CLK_tref - 1'b1))    
                                    ctrl_tref_cnt <= 'd0;
		else if (CTRL_TREF_EVENT)
                                    ctrl_tref_cnt <= ctrl_tref_cnt + 1'b1;
        else                        ctrl_tref_cnt <= ctrl_tref_cnt;                                       
    end
    // Time-multiplexed neuron counter
	always @(posedge CLK, posedge RST)
		if      (RST)                                   post_neur_cnt <= 'd0;
        else if ((state == WAIT) || (state ==POP_TSTEP))
                                                        post_neur_cnt <= 'd0;
		else if (post_neur_cnt_inc & (CTRL_NEUR_EVENT | CTRL_TSTEP_EVENT | CTRL_TREF_EVENT))  
                                                        post_neur_cnt <= post_neur_cnt + POST_NEUR_PARALLEL;
        else                                            post_neur_cnt <= post_neur_cnt;

        // Time-multiplexed neuron counter
	always @(posedge CLK, posedge RST)
		if      (RST)                                   pre_neur_cnt <= 'd0;
        else if (state == WAIT)                         pre_neur_cnt <= 'd0;
		else if (pre_neur_cnt_inc & (CTRL_NEUR_EVENT | CTRL_TREF_EVENT))   
                                                        pre_neur_cnt <= pre_neur_cnt + 1'b1;
        else                                            pre_neur_cnt <= pre_neur_cnt;        
    always @(posedge CLK or posedge RST)                                              
            if      (RST)                                T_step_cnt <= 'd0;                                 
            else if (state == TREF)                      T_step_cnt <= 'd0;                                
            else if (CTRL_TSTEP_EVENT_negedge)           T_step_cnt <= T_step_cnt + 1'b1;                            
            else                                         T_step_cnt <= T_step_cnt;
    // Output logic      
    always @(*) begin
        case(state)
        TREF : begin
            // sram关键控制信号
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;            
            CTRL_TSTEP_EVENT    = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            CTRL_SCHED_POP_N    = 1'b1;
            // To aer_out
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            CTRL_AEROUT_POP_NEUR  = 1'b0;
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            AERIN_ACK           = 1'b0;
            
            // 控制器神经元地址
            CTRL_POST_NEURON_ADDRESS = post_neur_cnt;
            CTRL_PRE_NEURON_ADDRESS  = pre_neur_cnt;
            CTRL_SYNARRAY_ADDR  = {pre_neur_cnt,post_neur_cnt[$clog2(POST_NEUR_PARALLEL) +: SYN_ARRAY_ADDR_WIDTH - PRE_NEUR_ADDR_WIDTH]};
            
            CTRL_TREF_EVENT     = 1'b1;
            CTRL_PRE_NEUR_CS    = 1'b1;
            CTRL_POST_NEUR_CS   = 1'b1;
            CTRL_SYNARRAY_CS    = 1'b1;
            // 每2个周期进行读post_ram、读写syna_ram、post_neur_cnt计数
            if (ctrl_tref_cnt == R_W_SRAM_CLK_tref - 1'b1) begin
                CTRL_SYNARRAY_WE    = 1'b1;
                post_neur_cnt_inc   = 1'b1;
            end else begin
                CTRL_SYNARRAY_WE    = 1'b0;
                post_neur_cnt_inc   = 1'b0;    
            end
            // 历遍完一轮突触后神经元后，pre_neru_cnt计数并更新突触前脉冲计数
            // 突触后计数为最后一组且拉高SRAM写信号时，更新突触前脉冲计数
            if ((post_neur_cnt == OUTPUT_NEURON - POST_NEUR_PARALLEL) && (ctrl_tref_cnt == R_W_SRAM_CLK_tref - 1'b1)) begin
                pre_neur_cnt_inc    = 1'b1;
                CTRL_PRE_NEUR_WE    = 1'b1;
            end
            else begin
                pre_neur_cnt_inc    = 1'b0;
                CTRL_PRE_NEUR_WE    = 1'b0;
            end

            // 当历遍突触前最后一个神经元时，每两个周期更新突触后脉冲计数
            if(IS_TRAIN) begin
                if ((ctrl_tref_cnt == R_W_SRAM_CLK_tref - 1'b1) && (pre_neur_cnt == INPUT_NEURON - 1'b1)) begin
                    CTRL_POST_NEUR_WE   = 1'b1;
                end else begin
                    CTRL_POST_NEUR_WE   = 1'b0;
                end
            end else begin
                if ((ctrl_tref_cnt == R_W_SRAM_CLK_tref - 1'b1)) begin
                    CTRL_POST_NEUR_WE   = 1'b1;
                end else begin
                    CTRL_POST_NEUR_WE   = 1'b0;
                end
            end

        end
        PUSH : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 控制器神经元地址
            CTRL_PRE_NEURON_ADDRESS = 'd0;
            CTRL_POST_NEURON_ADDRESS = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_POP_N    = 1'b1;
            // To aer_out
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            CTRL_AEROUT_POP_NEUR  = 1'b0;
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            AERIN_ACK           = 1'b0;
            post_neur_cnt_inc   = 1'b0;
            pre_neur_cnt_inc    = 1'b0;

            
            CTRL_SCHED_VIRTS    = AERIN_ADDR[AER_IN_WIDTH-1:AER_IN_WIDTH-2];
            CTRL_SCHED_ADDR     = AERIN_ADDR[PRE_NEUR_ADDR_WIDTH-1:0];// 神经元地址
            CTRL_SCHED_EVENT_IN = 1'b1;
        end

        NEUR_ACT : begin
            // To synaptic_core
            // sram关键控制信号
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 事件类型
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            CTRL_SCHED_POP_N    = 1'b1;
            // To aer_out
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            CTRL_AEROUT_POP_NEUR  = 1'b0;
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            AERIN_ACK           = 1'b0;
            pre_neur_cnt_inc    = 1'b0;

            // 控制器神经元地址
            CTRL_POST_NEURON_ADDRESS = post_neur_cnt;
            CTRL_PRE_NEURON_ADDRESS = SCHED_DATA_OUT[PRE_NEUR_ADDR_WIDTH-1:0];
            CTRL_SYNARRAY_ADDR  = {SCHED_DATA_OUT[PRE_NEUR_ADDR_WIDTH-1:0],CTRL_POST_NEURON_ADDRESS[$clog2(POST_NEUR_PARALLEL) +: SYN_ARRAY_ADDR_WIDTH - PRE_NEUR_ADDR_WIDTH]};
            CTRL_SYNARRAY_CS    = 1'b1;
            CTRL_SYNARRAY_WE    = 1'b0;
            CTRL_NEUR_EVENT     = 1'b1;
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b1;
            // 2个周期进行读写ram
            if (ctrl_neur_act_cnt == R_W_SRAM_CLK_neur_act - 1'b1) begin
                CTRL_POST_NEUR_WE  = 1'b1;
                post_neur_cnt_inc  = 1'b1;
            end else begin
                CTRL_POST_NEUR_WE  = 1'b0;
                post_neur_cnt_inc  = 1'b0;    
            end
        end

        POP_NEUR : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 事件类型
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            // To aer_out
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            CTRL_AEROUT_POP_NEUR  = 1'b0;
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            AERIN_ACK           = 1'b0;
            pre_neur_cnt_inc    = 1'b0;


            // 控制器神经元地址
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE  = 1'b0;
            CTRL_POST_NEURON_ADDRESS = 'd0;
            CTRL_PRE_NEURON_ADDRESS = SCHED_DATA_OUT[PRE_NEUR_ADDR_WIDTH-1:0];
            post_neur_cnt_inc       = 1'b0;

            // 注意！：由NEUR_ACT跳转至POP_NEUR时，ctrl_cnt为128或为偶数时以下逻辑才对，若不为偶数可将条件修改为ctrl_cnt==‘4计数
            // 增设ctrl_cnt清零条件，无需考虑ctrl_cnt计数
            // PRE_RAM读使能拉高两个时钟,写使能延后一个时钟拉高一时钟
            // SCHED_POP使能在第三个时钟拉低，第四个时钟跳转到WAIT
            CTRL_NEUR_EVENT     = 1'b1;
            CTRL_PRE_NEUR_CS    = (ctrl_cnt[1] == 1'b0)? 1'b1 : 1'b0;
            CTRL_PRE_NEUR_WE    = (ctrl_cnt[0] == 1'b0)? 1'b0 : 1'b1;
            CTRL_PRE_CNT_EN     = (ctrl_cnt[0] == 1'b0)? 1'b0 : 1'b1;
            CTRL_SCHED_POP_N    = (ctrl_cnt[1] == 1'b1)? 1'b0 : 1'b1;

        end
        TSTEP_ACT : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            CTRL_SCHED_POP_N    = 1'b1;
            // To aer_out
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            AERIN_ACK           = 1'b0;
            pre_neur_cnt_inc    = 1'b0;
            CTRL_PRE_NEURON_ADDRESS = 'd0;
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;

            // 控制器神经元地址
            CTRL_TSTEP_EVENT    = 1'b1;
            CTRL_AEROUT_POP_NEUR = 1'b1;
            CTRL_POST_NEURON_ADDRESS = post_neur_cnt;
            CTRL_POST_NEUR_CS   = 1'b1;
            
            // 2个周期进行读写ram
            if (ctrl_tstep_act_cnt == R_W_SRAM_CLK_tstep_act - 1'b1) begin
                CTRL_POST_NEUR_WE  = 1'b1;
                CTRL_AEROUT_PUSH_NEUR = 1'b1;
                post_neur_cnt_inc       = 1'b1;
            end else begin
                CTRL_POST_NEUR_WE  = 1'b0;
                CTRL_AEROUT_PUSH_NEUR = 1'b0;
                post_neur_cnt_inc       = 1'b0;    
            end
        end
        POP_NEUR_OUT : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            

            // 其他信号
            AERIN_ACK           = 1'b0;
            post_neur_cnt_inc   = 1'b0;
            pre_neur_cnt_inc    = 1'b0;
            CTRL_PRE_NEURON_ADDRESS = 'd0;
            CTRL_POST_NEURON_ADDRESS = 'd0;
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            // 控制器神经元地址
            // To aer_out
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            CTRL_AEROUT_POP_NEUR  = 1'b1;
            CTRL_SCHED_POP_N    = 1'b0;
        end
        
       POP_TSTEP : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_POP_N    = 1'b1;
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
        
            // 其他信号
            AERIN_ACK           = 1'b0;
            post_neur_cnt_inc   = 1'b0;
            pre_neur_cnt_inc    = 1'b0;
            CTRL_PRE_NEURON_ADDRESS = 'd0;
            CTRL_POST_NEURON_ADDRESS = 'd0;
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            // 控制器神经元地址
            // To aer_out
            CTRL_AEROUT_POP_TSTEP = 1'b1; 
            CTRL_AEROUT_POP_NEUR  = 1'b1;

        end
        WAIT_REQDN : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 控制器神经元地址
            CTRL_PRE_NEURON_ADDRESS = 'd0;
            CTRL_POST_NEURON_ADDRESS = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            CTRL_SCHED_POP_N    = 1'b1;
            // To aer_out
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            CTRL_AEROUT_POP_NEUR  = 1'b0;
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            post_neur_cnt_inc   = 1'b0;
            pre_neur_cnt_inc    = 1'b0; 
            AERIN_ACK           = 1'b1;
        end
        default : begin
            // To synaptic_core
            CTRL_SYNARRAY_ADDR  = 'd0;
            CTRL_SYNARRAY_CS    = 1'b0;
            CTRL_SYNARRAY_WE    = 1'b0;
            // sram关键控制信号
            CTRL_PRE_NEUR_CS    = 1'b0;
            CTRL_PRE_NEUR_WE    = 1'b0;
            CTRL_POST_NEUR_CS   = 1'b0;
            CTRL_POST_NEUR_WE   = 1'b0;
            CTRL_SYNA_WR_EVENT  = 1'b0;
            CTRL_SYNA_RD_EVENT  = 1'b0;
            CTRL_SYNA_PROG_DATA = 'd0;
            CTRL_PRE_CNT_EN     = 1'b0;
            // To neuron
            // SPI控制读写事件
            CTRL_WR_NEUR_EVENT  = 1'b0;
            CTRL_RD_NEUR_EVENT  = 1'b0;
            // SPI控制编入数据
            CTRL_POST_NEUR_PROG_DATA = 'd0;
            // 控制器神经元地址
            CTRL_PRE_NEURON_ADDRESS = 'd0;
            CTRL_POST_NEURON_ADDRESS = 'd0;
            // 事件类型
            CTRL_NEUR_EVENT     = 1'b0;
            CTRL_TSTEP_EVENT    = 1'b0;
            CTRL_TREF_EVENT     = 1'b0;
            // To scheduler
            CTRL_SCHED_VIRTS    = 'd0;
            CTRL_SCHED_ADDR     = 'd0;
            CTRL_SCHED_EVENT_IN = 1'b0;
            CTRL_SCHED_POP_N    = 1'b1;
            // To aer_out
            CTRL_AEROUT_PUSH_NEUR = 1'b0;
            CTRL_AEROUT_POP_NEUR  = 1'b0;
            CTRL_AEROUT_POP_TSTEP = 1'b0; 
            // 其他信号
            AERIN_ACK           = 1'b0;
            post_neur_cnt_inc   = 1'b0;
            pre_neur_cnt_inc    = 1'b0; 
        end 
        endcase
    end
endmodule


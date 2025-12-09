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
// Last modified Date:     2025/02/28 23:41:28
// Last Version:           V1.0
// Descriptions:           
//----------------------------------------------------------------------------------------
// Created by:             Sephiroth
// Created date:           2025/02/28 23:41:28
// mail      :             1245598043@qq.com
// Version:                V1.0
// TEXT NAME:              parallel_to_serial.v
// PATH:                   D:\BaiduSyncdisk\SNN_FFSTBP\rtl\axi_rw\parallel_to_serial.v
// Descriptions:   4位并行转串行模块
// 输入4位并行数据，输出1位串行数据，从低位开始输出（小端）        
//                         
//----------------------------------------------------------------------------------------
//****************************************************************************************//

module parallel_to_serial #(
    parameter                           DATA_WIDTH                 = 4     ,
    parameter                           CNT_MAX                    = 784    // INPUT_SIZE -1
)
(
    input                               CLK                        ,
    input                               rst_n                      ,
    input  wire        [DATA_WIDTH-1: 0]        din_parallel               ,
    input  wire                         din_valid                  ,
    input  wire                         AER_IN_ACK                 ,
    output reg                          pts_ready                  ,
    output reg         [  11: 0]        AER_IN_ADDR                ,
    output reg                          AER_IN_REQ                 ,
    output wire                         finish                      
);
    wire                                shift_en                    ;
    wire                                dout_serial                 ;
    wire                                AER_IN_REQ_negedge          ;
    wire                                tstep_valid_posedge         ;
    wire                                tstep_valid_negedge         ;
    wire                                tstep_start                 ;
    wire                                tstep_end                   ;

    reg                                 AER_IN_REQ_int              ;
    reg                                 tstep_valid                 ;
    reg                                 tstep_valid_int             ;
    reg                [DATA_WIDTH-1: 0]        din_parallel_tmp            ;
    reg                [  13: 0]        cnt                         ;
    reg                [   3: 0]        tstep_cnt                   ;

    assign                              AER_IN_REQ_negedge          = AER_IN_REQ_int && !AER_IN_REQ;
    assign                              tstep_valid_posedge         = tstep_valid && !tstep_valid_int;
    assign                              tstep_valid_negedge         = !tstep_valid && tstep_valid_int;
    // assign                              dout_serial                 = din_parallel_tmp[0];
    assign                              dout_serial                 = din_parallel_tmp[3];
    assign                              shift_en                    = !pts_ready && (AER_IN_REQ_negedge || !dout_serial) && !tstep_valid;
	assign 								finish                      = (tstep_cnt == 0) && tstep_valid_negedge;
    assign                              tstep_start                 = (cnt[9:0] == CNT_MAX - 1'b1) && shift_en;
    assign                              tstep_end                   = (tstep_valid && AER_IN_REQ_negedge);
    always @(posedge CLK or negedge rst_n) begin
        if(!rst_n) begin
            AER_IN_REQ_int <= 1'b0;
            tstep_valid_int <= 1'b0;
        end
        else begin
            AER_IN_REQ_int <= AER_IN_REQ;
            tstep_valid_int <= tstep_valid;
        end
    end
	
	// 计数器cnt\tstep_cnt
    always@(posedge CLK or negedge rst_n) begin
        if(!rst_n) begin
            cnt <= 'd0;
            tstep_cnt <= 'd0;
        end
        else if(!pts_ready && shift_en) begin
            cnt <= (cnt == CNT_MAX -1'b1)? 'd0 : cnt + 1'd1;
            tstep_cnt <= (cnt == CNT_MAX -1'b1)? tstep_cnt + 1'd1 : tstep_cnt;
        end
        else begin
            cnt <= cnt;
            tstep_cnt <= tstep_cnt;
        end
    end
	
	// tstep_valid信号
    always@(posedge CLK or negedge rst_n)begin
        if(!rst_n)begin
            tstep_valid <= 1'b0;
        end
        else if(tstep_end)begin
            tstep_valid <= 1'b0;
        end
        else if(tstep_start)begin
            tstep_valid <= 1'b1;
        end
        else begin
            tstep_valid <= tstep_valid;
        end
    end

	// pts_ready信号
    always @(posedge CLK or negedge rst_n) begin
        if(!rst_n || ((cnt[1:0] == 2'b11) && shift_en && !tstep_start) || tstep_end)
            pts_ready <= 1'b1;
        else if(pts_ready && din_valid)
            pts_ready <= 1'b0;
        else
            pts_ready <= pts_ready;
    end

	// din_parallel_temp 信号
    always@(posedge CLK or negedge rst_n)begin
        if(!rst_n)begin
            din_parallel_tmp <= 'd0;
        end
        else if((pts_ready && din_valid) && cnt[1:0] == 2'b00)begin
            din_parallel_tmp <= din_parallel;
        end
        else if( !pts_ready && shift_en)begin
            // din_parallel_tmp <= din_parallel_tmp >> 1;
            din_parallel_tmp <= din_parallel_tmp << 1;   
        end
        else begin
            din_parallel_tmp <= din_parallel_tmp;
        end
    end
	
	// Req信号
    always @(posedge CLK or negedge rst_n) begin
        if (!rst_n || AER_IN_ACK) begin
            AER_IN_REQ <= 1'b0;
        end
        else if(!AER_IN_REQ && !AER_IN_ACK && !pts_ready && (dout_serial || tstep_valid)) begin
              AER_IN_REQ <= 1'b1;
        end
        else
            AER_IN_REQ <= AER_IN_REQ;
    end


	// AER_IN_ADDR信号
    always @(posedge CLK or negedge rst_n)
        begin
            if(!rst_n)
                AER_IN_ADDR <= 12'b0;
            else if(tstep_valid_posedge || tstep_valid)
                AER_IN_ADDR <= {2'b01, cnt[9:0]};
            else
                AER_IN_ADDR <= {2'b00, cnt[9:0]};
        end
endmodule

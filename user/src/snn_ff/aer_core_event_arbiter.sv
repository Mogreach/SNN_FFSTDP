// 多核心仲裁轮询输出事件模块
module aer_core_event_arbiter #(
    parameter CORE_W        = 4,
    parameter CORE_H        = 4,
    parameter CORE_NUM      = 16,
    parameter AER_OUT_NEXT_LAYER_WIDTH = 12,
    parameter AER_OUT_CORE_WIDTH = 8  // 不含核心ID位
)(
    input  wire clk,
    input  wire rst,

    // ------------------ 来自每个 core 的输出 ------------------
    input  wire [CORE_NUM-1:0]                    core_req,   // 核心 FIFO 有有效事件
    input  wire [CORE_NUM*AER_OUT_CORE_WIDTH-1:0]      core_addr,  // 核心 FIFO 数据
    output reg  [CORE_NUM-1:0]                    core_ack,   // 核心 FIFO ACK

    // ------------------ 汇聚输出 ------------------
    output reg                                      evt_req,   // 输出事件请求
    output reg [AER_OUT_NEXT_LAYER_WIDTH-1:0]       evt_addr,  // 输出事件地址
    input  wire                                     evt_ack    // 输出 ACK
);
    // FSM 状态：轮询仲裁当前处理事件类型
    localparam ST_IDLE     = 2'd0;
    localparam ST_NEURON   = 2'd1;
    localparam ST_TIMESTEP = 2'd2;
    localparam CORE_ID_W = $clog2(CORE_NUM);
    // 事件类型
    localparam EVENT_NEURON   = 2'b00;
    localparam EVENT_TIMESTEP = 2'b01;
    reg [1:0] fsm_state;
    reg [1:0] fsm_next_state;
    
    // 轮询指针
    reg [$clog2(CORE_NUM)-1:0] cur_core;
    // 当前 core 的地址
    wire [AER_OUT_CORE_WIDTH-1:0] cur_core_event [0:CORE_NUM-1];
    // 当前轮询的 core 的信号
    wire [AER_OUT_CORE_WIDTH-1:0] cur_event = cur_core_event[cur_core];
    wire cur_req = core_req[cur_core];
    reg  cur_ack;
    reg  cur_ack_delay;
    // 事件类型
    wire [1:0] cur_event_type = cur_event[AER_OUT_CORE_WIDTH-1:AER_OUT_CORE_WIDTH-2];
    wire [AER_OUT_CORE_WIDTH-2-1:0] cur_event_id = cur_event[AER_OUT_CORE_WIDTH-2-1:0];
    wire no_req = !cur_req;
    wire neur_event_req = (cur_event_type == EVENT_NEURON) && cur_req;
    wire time_event_req = (cur_event_type == EVENT_TIMESTEP) && cur_req;

    wire cur_ack_negedge = cur_ack_delay && !cur_ack;
    wire cur_handshake_done = cur_ack_negedge;
    // 汇聚输出
    wire evt_hand_shake = evt_req && evt_ack;
    // 每个 core 轮询使能 ack信号，独热码
    reg [CORE_NUM-1:0] core_ack_en;
    // 每个core 接收tstep事件广播ack信号，
    reg [CORE_NUM-1:0] core_ack_tstep;
    always@(*) begin
        core_ack_en = ({{CORE_NUM-1{1'b0}},1'b1} << cur_core);
        core_ack_tstep =  ((fsm_state == ST_TIMESTEP) && (cur_core == CORE_NUM- 1'b1) && time_event_req) ? ({CORE_NUM{1'b1}}) : 'd0; //
    end
    
    genvar i;
    generate
        for (i = 0; i < CORE_NUM; i = i + 1) begin
            assign cur_core_event[i] = core_addr[i*AER_OUT_CORE_WIDTH+:AER_OUT_CORE_WIDTH];
            assign core_ack[i] = (core_ack_tstep[i] || core_ack_en[i]) && cur_ack;
        end
    endgenerate


    always @(posedge clk or posedge rst) begin
        if (rst) begin
            fsm_state <= ST_IDLE;
        end else begin
            fsm_state <= fsm_next_state;
        end
    end

    always @(*) begin 
        case (fsm_state)
            ST_IDLE: begin
                if(no_req)   fsm_next_state = ST_IDLE;
                else if(neur_event_req) fsm_next_state = ST_NEURON;
                else if(time_event_req) fsm_next_state = ST_TIMESTEP;
                else                    fsm_next_state = ST_IDLE;
            end
            ST_NEURON: begin
                if(no_req)   fsm_next_state = ST_IDLE;
                else if(neur_event_req) fsm_next_state = ST_NEURON;
                else if(time_event_req) fsm_next_state = ST_TIMESTEP;
                else                    fsm_next_state = ST_IDLE;
            end
            ST_TIMESTEP: begin
                if(no_req)   fsm_next_state = ST_IDLE;
                else if(neur_event_req) fsm_next_state = ST_NEURON;
                else if(time_event_req) fsm_next_state = ST_TIMESTEP;
                else                    fsm_next_state = ST_IDLE;  
            end
            default: begin
                fsm_next_state = ST_IDLE;
            end
        endcase
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cur_core <= 'd0;
        end else if((cur_event_type == EVENT_TIMESTEP) && (cur_core == CORE_NUM- 1'b1) && cur_handshake_done)begin
            cur_core <= 'd0;
        end else if(time_event_req && !(cur_core == CORE_NUM- 1'b1))begin
            cur_core <= cur_core + 1'b1;
        end
    end
    wire cur_ack_tstept_active = (cur_core == CORE_NUM- 1'b1) ? evt_ack : 1'b0;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cur_ack <= 'd0;
        end else if (fsm_state == ST_TIMESTEP)begin
            cur_ack <= cur_ack_tstept_active;
        end else begin
            cur_ack <= evt_ack;
        end
    end
    always @(posedge clk) begin
        cur_ack_delay <= cur_ack;
    end
    wire [$clog2(CORE_H)-1:0] core_h;
    wire [$clog2(CORE_W)-1:0] core_w;

    assign core_h = cur_core / CORE_W;
    assign core_w = cur_core % CORE_W;
    // Output
    always @(*) begin
        evt_req = cur_req;
        evt_addr = {cur_event_type,cur_event_id,core_h,core_w};
    end

    

endmodule

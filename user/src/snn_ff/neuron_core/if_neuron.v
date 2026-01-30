
module if_neuron #(
    parameter TIME_STEP = 8,
    parameter AER_WIDTH = 12,
    parameter POST_NEUR_MEM_WIDTH = 12,
    parameter POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter WEIGHT_WIDTH = 8
)( 
    input  wire CLK,
    input  wire  [POST_NEUR_SPIKE_CNT_WIDTH-1:0] post_spike_cnt,          // 突触后神经元发放脉冲数量 from SRAM
    output  wire [POST_NEUR_SPIKE_CNT_WIDTH-1:0] post_spike_cnt_next,          // 突触后神经元发放脉冲数量 to SRAM

    input  wire signed [POST_NEUR_MEM_WIDTH-1:0] param_thr,               // neuron firing threshold parameter 
    
    input  wire signed [POST_NEUR_MEM_WIDTH-1:0] state_core,              // core neuron state from SRAM 
    output wire signed [POST_NEUR_MEM_WIDTH-1:0] state_core_next,         // next core neuron state to SRAM
    
    input  wire signed [WEIGHT_WIDTH-1:0] syn_weight,              // synaptic weight
    input  wire                 neuron_event,               // synaptic event trigger
    input  wire                 time_step_event,
    input  wire                 time_ref_event,                // time reference event trigger
    input wire [clog2(TIME_STEP)-1:0] current_time_step,
    output reg                 spike_out                // neuron spike event output  
);
    localparam max_value = (1 << (POST_NEUR_MEM_WIDTH-1)) - 1;
    localparam min_value = -(1 << (POST_NEUR_MEM_WIDTH-1));

    reg signed [POST_NEUR_MEM_WIDTH-1:0] state_core_reg;
    reg signed [WEIGHT_WIDTH-1:0] syn_weight_reg;
    reg signed [POST_NEUR_MEM_WIDTH-1:0] param_thr_reg;
    //time_step_event：单时间步事件，待处理完一个时间步所有的神经元事件后发起，判断脉冲发放、膜电位复位、脉冲计数+1
    //time_ref_event: 一定时间步后拉高，重置脉冲计数以及更新权重（需要增加一个重置计数的信号）
    //neuron_event：神经元事件，只更新累加膜电位，以及输入神经元的脉冲数
    //core是膜电位数值，符号数，11位为符号位
    reg  [POST_NEUR_SPIKE_CNT_WIDTH-1:0] post_spike_cnt_next_i;
    reg  signed [POST_NEUR_MEM_WIDTH-1:0] state_core_next_i;
    wire signed [POST_NEUR_MEM_WIDTH-1:0] syn_weight_ext;
    wire signed [POST_NEUR_MEM_WIDTH-1:0] state_syn;

    assign state_core_next =  spike_out ? 'd0 : state_core_next_i;

    assign post_spike_cnt_next = post_spike_cnt_next_i;

    assign state_syn = state_core_reg + syn_weight_reg;
    assign overflow = (state_core_reg[POST_NEUR_MEM_WIDTH-1]==syn_weight_reg[WEIGHT_WIDTH-1]) && (state_syn[POST_NEUR_MEM_WIDTH-1]!=state_core_reg[POST_NEUR_MEM_WIDTH-1]);
    
    // One-hot encoding for current time step; cover all time steps
    wire  [TIME_STEP-1:0] time_one_hot_flag = ({{TIME_STEP-1{1'b0}},1'b1} << current_time_step);
    wire  [POST_NEUR_SPIKE_CNT_WIDTH-1:0] post_spike_cnt_next_ii = post_spike_cnt | time_one_hot_flag;
    
    always @(posedge CLK)           
    begin         
        state_core_reg <= state_core;                               
        syn_weight_reg <= syn_weight;
        param_thr_reg  <= param_thr;
    end                                          

    always @(*) begin 
        if (time_step_event) begin
            state_core_next_i = state_core;
            post_spike_cnt_next_i = state_core[POST_NEUR_MEM_WIDTH]? post_spike_cnt : post_spike_cnt_next_ii; // 膜电位大于0,标记当前时间步，设为1
            spike_out       = (state_core >= param_thr)? 1'b1: 1'b0;
        end
        else if (time_ref_event)begin 
            state_core_next_i = 'd0;
            post_spike_cnt_next_i = 'd0;
            spike_out = 1'b0;
        end
        else if (neuron_event) begin
            state_core_next_i = (overflow) ? 
            (state_syn[POST_NEUR_MEM_WIDTH-1])? max_value : min_value
                                                             : state_syn; //防止在一个时间步前，膜电位数值溢出变为负数，导致单个时间步内脉冲发放不了
            post_spike_cnt_next_i = post_spike_cnt;
            spike_out = 1'b0;
        end
        else begin 
            state_core_next_i = state_core;
            post_spike_cnt_next_i = post_spike_cnt;
            spike_out = 1'b0;
        end
    end
endmodule

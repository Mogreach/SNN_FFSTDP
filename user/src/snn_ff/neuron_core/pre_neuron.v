module pre_neuron 
#(
    parameter PRE_NEUR_SPIKE_CNT_WIDTH = 8,  // 突触前神经元脉冲计数数据位宽
    parameter TIME_STEP = 8
)
( 
    input  wire  [PRE_NEUR_SPIKE_CNT_WIDTH-1:0] pre_spike_cnt,          // 突触前神经元发放脉冲数量 from SRAM
    input  wire                 neuron_event,               // synaptic event trigger
    input  wire                 neuron_event_pulse,
    input  wire                 time_ref_event,                // time reference event trigger
    input wire [$clog2(TIME_STEP)-1:0] current_time_step,
    output reg [PRE_NEUR_SPIKE_CNT_WIDTH-1:0] pre_spike_cnt_next          // 突触前神经元发放脉冲数量 to SRAM
);
    //neuron_event：神经元事件，只更新累加膜电位，以及输入神经元的脉冲数
    //time_step_event：单时间步事件，待处理完一个时间步所有的神经元事件后发起，判断脉冲发放、膜电位复位、脉冲计数+1
    //time_ref_event: 一定时间步后拉高，重置脉冲计数以及更新权重（需要增加一个重置计数的信号）

    // One-hot encoding for current time step; cover all time steps
    wire  [TIME_STEP-1:0] time_one_hot_flag = ({{TIME_STEP-1{1'b0}},1'b1} << current_time_step);
    wire  [PRE_NEUR_SPIKE_CNT_WIDTH-1:0] pre_spike_cnt_next_i = pre_spike_cnt | time_one_hot_flag;

    always @(*) begin 
        if (neuron_event) begin
            pre_spike_cnt_next = pre_spike_cnt_next_i;
        end
        else if (time_ref_event)begin 
            pre_spike_cnt_next = 'd0;
        end
        else begin 
            pre_spike_cnt_next = pre_spike_cnt;
        end
    end
    


endmodule

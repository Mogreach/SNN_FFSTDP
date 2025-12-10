module pre_neuron ( 
    input  wire  [          7:0] pre_spike_cnt,          // 突触前神经元发放脉冲数量 from SRAM
    input  wire                 neuron_event,               // synaptic event trigger
    input  wire                 neuron_event_pulse,
    input  wire                 time_ref_event,                // time reference event trigger

    output  wire [          7:0] pre_spike_cnt_next          // 突触前神经元发放脉冲数量 to SRAM
);
    //neuron_event：神经元事件，只更新累加膜电位，以及输入神经元的脉冲数
    //time_step_event：单时间步事件，待处理完一个时间步所有的神经元事件后发起，判断脉冲发放、膜电位复位、脉冲计数+1
    //time_ref_event: 一定时间步后拉高，重置脉冲计数以及更新权重（需要增加一个重置计数的信号）
    reg  [7:0] pre_spike_cnt_next_i;
    assign pre_spike_cnt_next = pre_spike_cnt_next_i;
    always @(*) begin 
        if (neuron_event) begin
            pre_spike_cnt_next_i = (neuron_event_pulse)? pre_spike_cnt + 1: pre_spike_cnt;
        end
        else if (time_ref_event)begin 
            pre_spike_cnt_next_i = 'd0;
        end
        else begin 
            pre_spike_cnt_next_i = pre_spike_cnt;
        end
    end
    


endmodule

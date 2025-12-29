module if_neuron_tb;
    // 寄存器和线网
    reg [6:0] post_spike_cnt_reg;
    reg signed [11:0] state_core_reg;

    // 时钟信号
    reg clk;
    initial clk = 0;
    always #5 clk = ~clk;

    // 参数定义
    wire [6:0] post_spike_cnt;            // 输入脉冲计数
    reg signed [11:0] param_thr;         // 膜电位阈值
    wire signed [11:0] state_core;        // 神经元当前膜电位
    reg [7:0] syn_weight;                // 突触权重
    reg neuron_event;                    // 神经元事件
    reg time_step_event;                 // 时间步事件
    reg time_ref_event;                  // 时间参考事件
    
    wire spike_out;                      // 脉冲输出
    wire [6:0] post_spike_cnt_next;      // 输出脉冲计数
    wire signed [11:0] state_core_next;  // 输出膜电位


    assign post_spike_cnt = post_spike_cnt_reg;
    assign state_core = state_core_reg;

    // 实例化 if_neuron 模块
    if_neuron uut (
        .post_spike_cnt(post_spike_cnt),
        .post_spike_cnt_next(post_spike_cnt_next),
        .param_thr(param_thr),
        .state_core(state_core),
        .state_core_next(state_core_next),
        .syn_weight(syn_weight),
        .neuron_event(neuron_event),
        .time_step_event(time_step_event),
        .time_ref_event(time_ref_event),
        .spike_out(spike_out)
    );


    // 初始化信号
    initial begin
        // 初始设置
        post_spike_cnt_reg = 7'd0;
        param_thr = 12'd10;
        state_core_reg = 12'd5;
        neuron_event = 0;
        time_step_event = 0;
        time_ref_event = 0;

        // 模拟过程
        #10;  // 初始延时
        
        // 模拟一个时间步事件
        neuron_event = 1;
        #160;
        neuron_event = 0;
        
        // 模拟一个时间步事件触发后的状态更新
        time_step_event = 1;
        #10;
        time_step_event = 0;
        #10

        // 模拟一个时间参考事件触发后的脉冲计数重置
        time_ref_event = 1;
        #10;
        time_ref_event = 0;

        // 观察状态和输出
        $display("Final state_core_next: %d, post_spike_cnt_next: %d", state_core_next, post_spike_cnt_next);
        $finish;
    end

    // 时钟的正沿触发，更新输出
    always @(posedge clk) begin
        post_spike_cnt_reg <= post_spike_cnt_next;         // 对输出脉冲计数进行寄存
        state_core_reg <= state_core_next;                 // 对输出膜电位进行寄存
        syn_weight <= $random;  // 随机赋值，范围为32位整数
    end

    // 用于显示每个时钟周期的信息
    always @(posedge clk) begin
        $display("Time: %0t | state_core: %d, state_core_next: %d, post_spike_cnt: %d, post_spike_cnt_next: %d, spike_out: %d", 
                $time, state_core, state_core_next, post_spike_cnt, post_spike_cnt_next, spike_out);
    end

endmodule

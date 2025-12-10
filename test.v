`timescale 1ns/1ps

module state_machine_tb;

    // 测试平台信号
    reg  clk;
    reg  rst_n;
    wire [1:0] state;
    wire [1:0] state_next;
    

    // 实例化状态机模块
    state_machine uut (
        .clk(clk),
        .rst_n(rst_n),
        .state(state),
        .state_next(state_next)

    );

    // 时钟生成
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns 周期时钟
    end

    // 测试流程
    initial begin
        // 初始化
        rst_n = 0; // 复位
        #20;       // 等待 20ns
        rst_n = 1; // 释放复位

        // 观察状态转移
        #600;      // 运行 200ns
        $finish;     // 停止仿真
    end

    // 监视状态和计数器
    initial begin
        $monitor("Time: %0t | State: %b | Counter: %0d | Next_state： %b", $time, state, uut.counter,state_next);
    end
endmodule
module state_machine (
    input  wire clk,          // 时钟信号
    input  wire rst_n,        // 复位信号（低电平有效）
    output reg  [1:0] state,   // 当前状态输出
    output reg   [1:0] state_next 
);

    // 状态定义
    parameter [1:0] STATE_A = 2'b00;
    parameter [1:0] STATE_B = 2'b01;
    parameter [1:0] STATE_C = 2'b10;

    reg  [3:0] counter;       // 计数器
    wire       counter_lt_10; // 计数器是否小于 10

    // 计数器比较逻辑
    assign counter_lt_10 = (counter < 4'd10);
    always @(posedge clk or negedge rst_n)           
        begin                                        
            if(!rst_n)                               
                state <= STATE_A;                        
            else begin
                state <= state_next;
            end                                                                    
        end                                          
    // 状态转移逻辑（第一段：状态转移）
    always @(*) begin
        case (state)
            STATE_A: begin
                state_next = STATE_B; // 无条件转移到状态 B
            end
            STATE_B: begin
                state_next = STATE_C; // 无条件转移到状态 C
            end
            STATE_C: begin
                if (counter_lt_10) begin
                    state_next = STATE_A; // 如果计数器小于 10，转移到状态 A
                end
                else begin
                    state_next = STATE_C; // 否则停留在状态 C
                end
            end
            default: begin
                state_next = STATE_A; // 默认转移到状态 A
            end
        endcase
    end

    // 计数器逻辑（第二段：状态行为）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 4'b0; // 复位时计数器清零
        end
        else if (state == STATE_B) begin
            counter <= counter + 1; // 在状态 B 时计数器加 1
        end
    end

    // 输出逻辑（第三段：状态输出）
    // 这里状态直接输出，可以根据需要添加其他输出逻辑
    // 例如：assign out_signal = (state == STATE_C);

endmodule

`timescale 1ns / 1ps
module Top
(
    input  wire                         CLK                        ,
    input  wire                         RST                        ,
    input  wire        [ 11: 0]         AERIN_ADDR                 ,
    input  wire                         AERIN_REQ                  ,
    input  wire                         IS_POS                     ,
    input  wire                         IS_TRAIN                   ,
    output wire                         AERIN_ACK                  ,
    output wire        [  31: 0]        GOODNESS                   ,
    output wire                         PROCESS_DONE
);
    // 全局参数
    parameter TIME_STEP = 8; // 时间步长
    parameter WEIGHT_WIDTH = 9; // 
    genvar L1_NUM;
    // 例化多个3*3*3 -> 4*1的模块
    generate
        for (L1_NUM=0; L1_NUM < 100; L1_NUM=L1_NUM+1)begin : gen_core
            localparam INPUT_NEURON = 27;
            localparam OUTPUT_NEURON = 4;
            
            localparam AER_WIDTH = 2 + log2(INPUT_NEURON); // AER地址宽度, 至少2 + log2(输入神经元数)

            localparam POST_NEUR_PARALLEL = 4; //并行度       
            localparam PRE_NEUR_WORD_ADDR_WIDTH = $clog2(INPUT_NEURON) + 1; // log2（输入神经元数）+ 1
            localparam PRE_NEUR_BYTE_ADDR_WIDTH = 0; // 无用参数，保持0

            localparam POST_NEUR_ADDR_WIDTH = $clog2(OUTPUT_NEURON) + 1; // log2（输出神经元数）+ 1
            localparam POST_NEUR_BYTE_ADDR_WIDTH = $clog2(POST_NEUR_PARALLEL); // 并行输出神经元字节地址位宽
            localparam POST_NEUR_WORD_ADDR_WIDTH= POST_NEUR_ADDR_WIDTH - POST_NEUR_BYTE_ADDR_WIDTH; // 输出神经元字地址位宽

            localparam SYN_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH; // 突触阵列数据位宽
            localparam SYN_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL); // 突触阵列地址位宽 

            // 神经元状态数据位宽
            parameter PRE_NEUR_DATA_WIDTH = $clog2(TIMT_NEE_STEP) + 1, // 单个突触前神经元脉冲计数数据位宽

            parameter POST_NEUR_RELU_CNT_WIDTH = $clog2(TIME_STEP) + 1,// 单个突触后神经元Relu(膜电位大于0)计数器数据位宽
            parameter POST_NEUR_MEM_WIDTH = 13, // 单个突触后神经元膜电位数据位宽
            parameter POST_NEUR_SPIKE_CNT_WIDTH = 6, // 单个突触后神经元脉冲计数数据位宽

            // 单个突触后神经元状态数据位宽 {en, POST_NEUR_RELU_CNT_WIDTH, POST_NEUR_SPIKE_CNT_WIDTH, POST_NEUR_MEM_WIDTH}
            parameter POST_NEUR_DATA_WIDTH = 1 + POST_NEUR_RELU_CNT_WIDTH + POST_NEUR_SPIKE_CNT_WIDTH + POST_NEUR_MEM_WIDTH, 


            wire               [AER_WIDTH-1: 0] AEROUT_ADDR                 ;
            wire                                AEROUT_REQ                  ;
            wire                                AEROUT_ACK                  ;
            reg                                 AEROUT_ACK_reg              ;
            reg                [   5: 0]        AEROUT_ACK_delay            ;
            
            wire                                ONE_SAMPLE_FINISH           ;
            wire                                SCHED_FULL                  ;
            
            assign PROCESS_DONE = ONE_SAMPLE_FINISH;
            ODIN_ffstdp#(
                .TIME_STEP                             (TIME_STEP          ),
                .INPUT_NEURON                          (INPUT_NEURON       ),
                .OUTPUT_NEURON                         (OUTPUT_NEURON      ),
                .AER_WIDTH                             (AER_WIDTH          ),
                .PRE_NEUR_ADDR_WIDTH                   (PRE_NEUR_ADDR_WIDTH),
                .PRE_NEUR_WORD_ADDR_WIDTH              (PRE_NEUR_WORD_ADDR_WIDTH),
                .PRE_NEUR_BYTE_ADDR_WIDTH              (PRE_NEUR_BYTE_ADDR_WIDTH),
                .POST_NEUR_ADDR_WIDTH                  (POST_NEUR_ADDR_WIDTH),
                .POST_NEUR_WORD_ADDR_WIDTH             (POST_NEUR_WORD_ADDR_WIDTH),
                .POST_NEUR_BYTE_ADDR_WIDTH             (POST_NEUR_BYTE_ADDR_WIDTH),
                .POST_NEUR_PARALLEL                    (POST_NEUR_PARALLEL ),
                .PRE_NEUR_DATA_WIDTH                   (PRE_NEUR_DATA_WIDTH),
                .POST_NEUR_DATA_WIDTH                  (POST_NEUR_DATA_WIDTH),
                .POST_NEUR_MEM_WIDTH                   (POST_NEUR_MEM_WIDTH),
                .POST_NEUR_SPIKE_CNT_WIDTH             (POST_NEUR_SPIKE_CNT_WIDTH),
                .SYN_ARRAY_DATA_WIDTH                  (SYN_ARRAY_DATA_WIDTH),
                .SYN_ARRAY_ADDR_WIDTH                  (SYN_ARRAY_ADDR_WIDTH),
                .WEIGHT_WIDTH                          (WEIGHT_WIDTH       )  
            )
            u_ODIN_ffstdp(
            // Global input     -------------------------------
                .CLK                                (CLK                       ),
                .RST                                (RST                       ),
                .IS_POS                             (IS_POS                    ),// 0: negative, 1: positive
                .IS_TRAIN                           (IS_TRAIN                  ),// 0: inference, 1: training
            // Input 12-bit AER -------------------------------
                .AERIN_ADDR                         (AERIN_ADDR                ),
                .AERIN_REQ                          (AERIN_REQ                 ),
                .AERIN_ACK                          (AERIN_ACK                 ),
            // Output 10-bit AER -------------------------------
                .AEROUT_ADDR                        (AEROUT_ADDR               ),
                .AEROUT_REQ                         (AEROUT_REQ                ),
                .AEROUT_ACK                         (AEROUT_ACK                ),
                .GOODNESS                           (GOODNESS                  ),
                .ONE_SAMPLE_FINISH                  (ONE_SAMPLE_FINISH         )
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
        end
    endgenerate
                                                                    
endmodule
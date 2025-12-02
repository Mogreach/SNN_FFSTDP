module neuron_core #(
    parameter N = 784,
    parameter M = 8
)(
    // Global inputs ------------------------------------------
    input  wire CLK,
    input  wire RST_N,
    // Synaptic inputs ----------------------------------------
    input  wire [         31:0] SYNARRAY_RDATA,
    // Controller inputs ----------------------------------------
        // SPI控制编入数据
    input wire  [31:0] CTRL_POST_NEUR_PROG_DATA,
        //控制器突触地址
    input wire [9:0] CTRL_PRE_NEURON_ADDRESS,
    input wire [9:0] CTRL_POST_NEURON_ADDRESS,
        //SPI控制读写事件
    input wire       CTRL_WR_NEUR_EVENT,
    input wire       CTRL_RD_NEUR_EVENT,
        //训练推理事件
    input wire  CTRL_NEUR_EVENT,
    input wire  CTRL_TSTEP_EVENT,
    input wire  CTRL_TREF_EVENT,

    input wire  CTRL_PRE_NEUR_CS,
    input wire  CTRL_PRE_NEUR_WE,
    input wire  CTRL_POST_NEUR_CS,
    input wire  CTRL_POST_NEUR_WE,
    input wire  CTRL_PRE_CNT_EN,
    // SPI inputs
    input wire  SPI_GATE_ACTIVITY_sync,
    input wire  [9:0] SPI_POST_NEUR_ADDR,

    // Outputs
    output reg [31:0] NEUR_STATE,
    output wire [3:0] NEUR_EVENT_OUT,
    output wire [7:0] PRE_NEUR_S_CNT,
    output wire [6:0] POST_NEUR_S_CNT_0,
    output wire [6:0] POST_NEUR_S_CNT_1,
    output wire [6:0] POST_NEUR_S_CNT_2,
    output wire [6:0] POST_NEUR_S_CNT_3
);
    // localparam  neuron_thresold= 12'b0_00001_100000; //神经元阈值S5.6: 1.5
    localparam  neuron_thresold= 12'b0000_0100_1101; // 神经元阈值S5.6: 1.2
    // localparam  neuron_thresold= 12'b0_00001_000000; //神经元阈值S5.6: 1
    // Internal regs and wires definitions
    wire [7:0] pre_neuron_sram_out;
    wire [7:0] pre_neuron_sram_in;

    wire [127:0] post_neuron_sram_out;
    wire [127:0] post_neuron_sram_in;
    wire [5:0]  post_neuron_sram_addr;
    wire [1:0]  post_neuron_byte_addr;
    wire [6:0]  post_neuron_index_addr;
    wire [3:0]  IF_neuron_event_out;

    wire [11:0] IF_neuron_next_mem [3:0];
    wire [6:0]  IF_neuron_next_spike_cnt[3:0];
    wire [6:0]  POST_NEUR_S_CNT[3:0];

    // assign NEUR_STATE = (post_neuron_sram_out >> ({5'b0,post_neuron_byte_addr} << 5))[31:0]; //右移post_neuron_byte_address * 32
    assign post_neuron_sram_addr = CTRL_POST_NEURON_ADDRESS[7:2];
    assign post_neuron_byte_addr = CTRL_POST_NEURON_ADDRESS[1:0];
    assign PRE_NEUR_S_CNT = pre_neuron_sram_out;
    assign POST_NEUR_S_CNT_0 = POST_NEUR_S_CNT[0];
    assign POST_NEUR_S_CNT_1 = POST_NEUR_S_CNT[1];
    assign POST_NEUR_S_CNT_2 = POST_NEUR_S_CNT[2];
    assign POST_NEUR_S_CNT_3 = POST_NEUR_S_CNT[3];
    always @(*) begin
        case (post_neuron_byte_addr)
            2'b00: NEUR_STATE = post_neuron_sram_out[31:0];    // Output lower 32 bits
            2'b01: NEUR_STATE = post_neuron_sram_out[63:32];   // Output 32 bits from [63:32]
            2'b10: NEUR_STATE = post_neuron_sram_out[95:64];   // Output 32 bits from [95:64]
            2'b11: NEUR_STATE = post_neuron_sram_out[127:96];  // Output upper 32 bits
            default: NEUR_STATE = 32'b0;  // Default case
        endcase
    end

    // reg neur_event_d;
    // always @(posedge CLK or negedge RST_N)           
    //     begin                                        
    //         if(!RST_N)                               
    //             neur_event_d <= 0;                             
    //         else                                    
    //             neur_event_d <= CTRL_NEUR_EVENT; 
    //     end                                          

    pre_neuron pre_neuron_0( 
    .pre_spike_cnt(pre_neuron_sram_out),          // 突触前神经元发放脉冲数量 from SRAM
    .neuron_event(CTRL_NEUR_EVENT),               // synaptic event trigger
    .neuron_event_pulse(CTRL_PRE_CNT_EN),
    .time_ref_event(CTRL_TREF_EVENT),                // time reference event trigger
    .pre_spike_cnt_next(pre_neuron_sram_in)          // 突触前神经元发放脉冲数量 to SRAM
);

    genvar i;
    // 神经元状态更新模块 + SPI初始化
    generate
        for (i=0; i<4; i=i+1) begin
            // 神经元状态信息更新：SPI 配置？（SPI指定地址？掩码后的编入数据：保持）：膜电位更新
            // 突触后神经元发放脉冲更新
            // assign post_neuron_sram_in[30+(i*32):24+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[30:24] : IF_neuron_next_spike_cnt[i];
            assign post_neuron_sram_in[30+(i*32):24+(i*32)] =  IF_neuron_next_spike_cnt[i];
            // 突触后神经元阈值更新
            // assign post_neuron_sram_in[23+(i*32):12+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[23:12] : neuron_thresold;
            assign post_neuron_sram_in[23+(i*32):12+(i*32)] =  neuron_thresold;
            // 突触后神经元膜电位更新
            // assign post_neuron_sram_in[11+(i*32): 0+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[11:0]: IF_neuron_next_mem[i]; 
            assign post_neuron_sram_in[11+(i*32): 0+(i*32)] = IF_neuron_next_mem[i]; 
            // 突触后神经元使能信号更新                 
            // assign post_neuron_sram_in[31+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[31] : post_neuron_sram_out[31+(i*32)];                            
            // assign post_neuron_sram_in[31+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[31] : 1'b1;
            assign post_neuron_sram_in[31+(i*32)] = 1'b1;

            if_neuron if_neuron_gen( 
            .post_spike_cnt(post_neuron_sram_out [30+(i*32):24+(i*32)]),          // 突触后神经元发放脉冲数量 from SRAM
            .post_spike_cnt_next(IF_neuron_next_spike_cnt[i]),          // 突触后神经元发放脉冲数量 to SRAM
            .param_thr(post_neuron_sram_out [23+(i*32):12+(i*32)]),               // neuron firing threshold parameter 
            .state_core(post_neuron_sram_out [11+(i*32): 0+(i*32)]),              // core neuron state from SRAM 
            .state_core_next(IF_neuron_next_mem[i]),         // next core neuron state to SRAM
            .syn_weight(SYNARRAY_RDATA [7+(i*8):0+(i*8)]),              // synaptic weight
            .neuron_event(CTRL_NEUR_EVENT),                  // synaptic event trigger
            .time_step_event(CTRL_TSTEP_EVENT),
            .time_ref_event(CTRL_TREF_EVENT),                // time reference event trigger
            .spike_out(IF_neuron_event_out[i])                // neuron spike event output  
            );
            assign NEUR_EVENT_OUT[i] = post_neuron_sram_out[31+(i*32)] ? ((CTRL_POST_NEUR_CS && CTRL_POST_NEUR_WE) ? IF_neuron_event_out[i] : 1'b0) : 1'b0;
            assign POST_NEUR_S_CNT[i] = post_neuron_sram_out [30+(i*32):24+(i*32)];
        end
    endgenerate
    // 突触前神经元SRAM 读优先时序
    SRAM_1024x8_wrapper neurarray_pre (       
        
        // Global inputs
        .clk         (CLK),
    
        // Control and data inputs
        // .ena        (CTRL_PRE_NEUR_CS),
        .we         (CTRL_PRE_NEUR_WE),
        .a          (CTRL_PRE_NEURON_ADDRESS),
        .d          (pre_neuron_sram_in),
        // Data output
        .spo          (pre_neuron_sram_out)
    );

    // 突触后神经元SRAM 读优先时序
    SRAM_256x128_wrapper neurarray_post (       
        
        // Global inputs
        .clka         (CLK),
    
        // Control and data inputs
        .ena         (CTRL_POST_NEUR_CS),
        .wea         (CTRL_POST_NEUR_WE),
        .addra          (post_neuron_sram_addr),
        .dina          (post_neuron_sram_in),
        // Data output
        .douta          (post_neuron_sram_out)
    );

//    Neuron_SRAM neurarray_0(
//    .clka  (CLK      ),  // input wire clka
//    .ena   (CTRL_NEURMEM_CS       ),  // input 片选使能信号
//    .wea   (CTRL_NEURMEM_WE       ),  // input 写使能信号
//    .addra (CTRL_NEURMEM_ADDR     ), 
//    .dina  (neuron_data  ),
//    .douta (NEUR_STATE  )  
//);

endmodule




module SRAM_256x32_wrapper (

    // Global inputs
    input          CK,                       // Clock (synchronous read/write)

    // Control and data inputs
    input          CS,                       // Chip select
    input          WE,                       // Write enable
    input  [  7:0] A,                        // Address bus 
    input  [ 31:0] D,                        // Data input bus (write)

    // Data output
    output [ 31:0] Q                         // Data output bus (read)   
);
    /*
     *  Simple behavioral code for simulation, to be replaced by a 256-word 32-bit SRAM macro 
     *  or Block RAM (BRAM) memory with the same format for FPGA implementations.
     */      
        reg [31:0] SRAM[255:0];
        reg [31:0] Qr;
        always @(posedge CK) begin
            Qr <= CS ? SRAM[A] : Qr;
            if (CS & WE) SRAM[A] <= D;
        end
        assign Q = Qr;


endmodule

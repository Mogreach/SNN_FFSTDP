// `timescale 1ns/1ps

module tb_top_lrf_odins();

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam FM_W    = 16;
    localparam FM_H    = 16;
    localparam FM_C    = 3;
    localparam CORE_W  = 8;
    localparam CORE_H  = 8;
    localparam CORE_C  = 8;
    localparam LRF_W   = 3;
    localparam LRF_H   = 3;
    localparam TIME_STEP = 8;
    localparam POST_NEUR_MEM_WIDTH = 13;
    localparam WEIGHT_WIDTH = 9;
    localparam GOODNESS_WIDTH = 20;

    localparam CLK_PERIOD = 4;
    localparam AERIN_WIDTH = 2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W);
    localparam AEROUT_WIDTH = 2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H);


    // ------------------------------------------------------------
    // Signals
    // ------------------------------------------------------------
    logic clk;
    logic rst;
    logic IS_POS;
    logic IS_TRAIN;

    logic AERIN_REQ;
    logic [2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W)-1:0] AERIN_ADDR;
    logic AERIN_ACK;

    logic AEROUT_REQ;
    logic [2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H) - 1:0] AEROUT_ADDR;
    logic AEROUT_ACK;

    logic [GOODNESS_WIDTH-1:0] GOODNESS;

    // 模拟输出神经元事件
    logic [11:0] aer_neur_spk;
    logic auto_ack_verbose;

    // ------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------
    top_lrf_odins #(
        .FM_W(FM_W),
        .FM_H(FM_H),
        .FM_C(FM_C),
        .CORE_W(CORE_W),
        .CORE_H(CORE_H),
        .CORE_C(CORE_C),
        .LRF_W(LRF_W),
        .LRF_H(LRF_H),
        .TIME_STEP(TIME_STEP),
        .POST_NEUR_MEM_WIDTH(POST_NEUR_MEM_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .GOODNESS_WIDTH(GOODNESS_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .IS_POS(IS_POS),
        .IS_TRAIN(IS_TRAIN),
        .AERIN_REQ(AERIN_REQ),
        .AERIN_ADDR(AERIN_ADDR),
        .AERIN_ACK(AERIN_ACK),
        .AEROUT_REQ(AEROUT_REQ),
        .AEROUT_ADDR(AEROUT_ADDR),
        .AEROUT_ACK(AEROUT_ACK)
    );

    // ------------------------------------------------------------
    // Clock
    // ------------------------------------------------------------
    always #(CLK_PERIOD/2) clk = ~clk;
    // ------------------------------------------------------------
    // Test stimulus
    // ------------------------------------------------------------
    localparam int FM_HW_BITS = $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W);
    logic [FM_HW_BITS-1:0] pix_idx;
    initial begin
        clk = 0;
        rst = 1;
        IS_POS = 0;
        IS_TRAIN = 0;
        AERIN_ADDR = 0;
        AERIN_REQ = 0;
        AEROUT_ACK = 0;
        auto_ack_verbose = 1'b1;
        fork
        auto_ack(.req(AEROUT_REQ), .ack(AEROUT_ACK), .addr(AEROUT_ADDR), .neur(aer_neur_spk), .verbose(auto_ack_verbose));
        join_none
        #(2*CLK_PERIOD);
        rst = 0;
        // 模拟几个事件发送到 DUT
        #(3*CLK_PERIOD);

        IS_TRAIN = 1;
        IS_POS = 1;
        wait_ns(20);

        for (int y = 0; y < 16; y++) begin
            for (int x = 0; x < 16; x++) begin
                pix_idx = y * FM_W + x;

                aer_send(
                    {2'b00, pix_idx},   // neuron spike
                    AERIN_ADDR,
                    AERIN_ACK,
                    AERIN_REQ
                );
                wait_ns(10);
            end
        end
        aer_send({2'b01, pix_idx}, AERIN_ADDR, AERIN_ACK, AERIN_REQ);
                for (int y = 0; y < 16; y++) begin
            for (int x = 0; x < 16; x++) begin
                pix_idx = y * FM_W + x;

                aer_send(
                    {2'b00, pix_idx},   // neuron spike
                    AERIN_ADDR,
                    AERIN_ACK,
                    AERIN_REQ
                );
                wait_ns(10);
            end
        end
        wait_ns(20000);

        $display("GOODNESS: %h", GOODNESS);

        #(10*CLK_PERIOD);
        $display("Simulation completed at time %0t", $time);
        $finish;
    end




    // ------------------------------------------------------------
    // Stimulus task
    // ------------------------------------------------------------
    task automatic aer_send(
        input  logic [AERIN_WIDTH-1:0] addr_in,
        ref    logic [AERIN_WIDTH-1:0] addr_out,
        ref    logic                    ack,
        ref    logic                    req
    );
        while (ack) wait_ns(1);
        addr_out = addr_in;
        wait_ns(5);
        req = 1'b1;
        while (!ack) wait_ns(1);
        wait_ns(5);
        req = 1'b0;
        // 等待 ack 拉低（完整握手）
        while (ack) wait_ns(1);
    endtask


    task automatic auto_ack(
        ref logic req,
        ref logic ack,
        ref logic [AEROUT_WIDTH-1:0] addr,
        ref logic [11:0] neur,
        ref logic verbose
    );
        forever begin
            while (~req) wait_ns(1);
            wait_ns(50);
            neur = addr;
            if (verbose)
                $display("----- NEURON OUTPUT SPIKE (FROM AER): Event 0x%h at time %0t", neur, $time);
            ack = 1'b1;
            while (req) wait_ns(1);
            wait_ns(50);
            ack = 1'b0;
        end
    endtask


    task wait_ns;
        input integer tics_ns;
        #tics_ns;
    endtask


endmodule
// `timescale 1ns / 1ps
// module tb_top_lrf_odins_test();
// // ------------------------------------------------------------
// // Parameters
// // ------------------------------------------------------------
// localparam FM_W    = 28;
// localparam FM_H    = 28;
// localparam FM_C    = 1;
// localparam CORE_W  = 14;
// localparam CORE_H  = 14;
// localparam CORE_C  = 4;
// localparam LRF_W   = 3;
// localparam LRF_H   = 3;
// localparam TIME_STEP = 8;
// localparam POST_NEUR_MEM_WIDTH = 13;
// localparam WEIGHT_WIDTH = 9;
// localparam GOODNESS_WIDTH = 20;

// localparam CLK_PERIOD = 4;
// localparam AERIN_WIDTH = 2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W);
// localparam AEROUT_WIDTH = 2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H);

//   // Signals
//   logic CLK;
//   logic RST;
//   logic IS_POS;
//   logic IS_TRAIN;


//   logic [15:0] cnt;
//   logic [15:0] pixel_index;
// //   logic [7:0] aer_neur_spk;
//   logic ONE_SAMPLE_FINISH;
//   logic SCHED_FULL;
//   logic auto_ack_verbose;
//     // ------------------------------------------------------------
//     // Signals
//     // ------------------------------------------------------------
//     logic AERIN_REQ;
//     logic [AERIN_WIDTH-1:0] AERIN_ADDR;
//     logic AERIN_ACK;

//     logic AEROUT_REQ;
//     logic [AEROUT_WIDTH-1:0] AEROUT_ADDR;
//     logic AEROUT_ACK;
//     logic [GOODNESS_WIDTH-1:0] GOODNESS;
//     // 模拟输出神经元事件
//     logic [AERIN_WIDTH-1:0] aer_neur_spk;

//     // Instantiate the DUT (Device Under Test)
//     // ------------------------------------------------------------
//     // DUT
//     // ------------------------------------------------------------
//     top_lrf_odins #(
//         .FM_W(FM_W),
//         .FM_H(FM_H),
//         .FM_C(FM_C),
//         .CORE_W(CORE_W),
//         .CORE_H(CORE_H),
//         .CORE_C(CORE_C),
//         .LRF_W(LRF_W),
//         .LRF_H(LRF_H),
//         .TIME_STEP(TIME_STEP),
//         .POST_NEUR_MEM_WIDTH(POST_NEUR_MEM_WIDTH),
//         .WEIGHT_WIDTH(WEIGHT_WIDTH),
//         .GOODNESS_WIDTH(GOODNESS_WIDTH)
//     ) dut (
//         .clk(clk),
//         .rst(rst),
//         .IS_POS(IS_POS),
//         .IS_TRAIN(IS_TRAIN),
//         .AERIN_REQ(AERIN_REQ),
//         .AERIN_ADDR(AERIN_ADDR),
//         .AERIN_ACK(AERIN_ACK),
//         .AEROUT_REQ(AEROUT_REQ),
//         .AEROUT_ADDR(AEROUT_ADDR),
//         .AEROUT_ACK(AEROUT_ACK)
//     );

//   assign AEROUT_ADDR = dut.AEROUT_ADDR;
//   assign AEROUT_REQ = dut.AEROUT_REQ;
//   assign AEROUT_ACK = dut.AEROUT_ACK;
//   // Clock generation
//   initial begin
//     CLK = 0;
//     forever #(CLK_PERIOD / 2) CLK = ~CLK;
//   end
//   always @(posedge CLK) begin
//     if (cnt == 784)
//       cnt <= 0;
//     else if(AERIN_REQ && AERIN_ACK)
//       cnt <= cnt + 1;
//     else
//       cnt <= cnt;
//   end

//     // **读取 TXT 文件**
//     parameter int N = 1000;   // 样本数
//     parameter int T = 8;    // 时间步
//     parameter int WIDTH = 784;  // 每个时间步的 bit 数
    
//     bit spike_data_reshaped [0:N-1][0:T-1][0:WIDTH-1]; // 存储展开后的数据
//     integer file, byte_count;
//     bit [7:0] spike_byte;
//     integer bit_index = 0;
//     integer n_idx = 0, t_idx = 0, w_idx = 0;
//   initial begin
//       // file = $fopen("D:/BaiduSyncdisk/SNN_FFSTBP/sim/python/simulation_spikes.bin", "rb"); // 以二进制方式读取
//       // file = $fopen("D:/BaiduSyncdisk/SNN_FFSTBP/sim/python/all_spikes.bin", "rb"); // 以二进制方式读取
//       file = $fopen("D:/WorkSpace/Temporary/SNN_FFSTDP/user/data/all_spikes.bin", "rb");
      
//       if (file == 0) begin
//           $display("Error: Cannot open file!");
//           $finish;
//       end

//       // 读取所有数据
//       while (!$feof(file) && n_idx < N) begin
//             byte_count = $fread(spike_byte, file); // 读取 1 字节（8-bit）

//             if (byte_count > 0) begin
//                 for (int i = 7; i >= 0; i--) begin
//                     spike_data_reshaped[n_idx][t_idx][w_idx] = spike_byte[i]; // 存入数组
//                     // 更新索引
//                     w_idx++;
//                     if (w_idx == WIDTH) begin
//                         w_idx = 0;
//                         t_idx++;
//                         if (t_idx == T) begin
//                             t_idx = 0;
//                             n_idx++;
//                         end
//                     end
//                     if (n_idx >= N) break; // 读取到 N 个样本后停止
//                 end
//             end
//         end
//         $fclose(file);
//         $display("Spike data loaded successfully!");
//   end




//   always @(posedge CLK) begin
//     if (ONE_SAMPLE_FINISH)
//       IS_POS <= ~IS_POS;
//     else
//       IS_POS <= IS_POS;
//   end

//   // Reset and stimulus
//   initial begin                 
//   auto_ack_verbose = 1'b1;
//     fork
//       auto_ack(.req(AEROUT_REQ), .ack(AEROUT_ACK), .addr(AEROUT_ADDR), .neur(aer_neur_spk), .verbose(auto_ack_verbose));
//     join_none
//     // Initialize signals
//     RST = 1;
//     IS_POS = 0;
//     IS_TRAIN = 0;
//     AERIN_ADDR = 'd0;
//     AERIN_REQ = 0;
//     AEROUT_ACK = 0;
//     cnt = 0;

//     // Apply reset
//     #20;
//     RST = 0;

//     // Stimulus
//     #20;
//     IS_TRAIN = 0;
//     IS_POS = 1;
    
//     // 遍历 N 个样本，每个样本有 T 个时间步
//     for (int n = 0; n < N; n++) begin
//         int sample_index;
//         int time_index;
//         int pixel_index;
//         for (int t = 0; t < T; t++) begin
//             for (int pix = 0; pix < 784; pix++) begin
//                 sample_index = n;  // 选择当前样本
//                 time_index = t;    // 选择当前时间步
//                 pixel_index = pix; // 选择当前像素
//                 if (spike_data_reshaped[sample_index][time_index][pixel_index] == 1) begin
//                   aer_send (.addr_in({1'b0, 1'b0, pixel_index[AERIN_WIDTH-2-1:0]}), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
//                   wait_ns(10);
//                 end
//             end
//             aer_send (.addr_in({1'b0,1'b1,{AERIN_WIDTH-2{1'b1}}}), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
//             wait_ns(10);
//         end
        
//         if (n >= 20-1) begin
//             IS_TRAIN = 0;
//         end
//         $display("GOODNESS: %h", GOODNESS);
//     end
//     // Check results
    

//     // Finish simulation
//     #200;
//     $finish;
//   end

//   // Monitor signals
//   //initial begin
//   //  $monitor("Time: %0t, AERIN_ADDR: %h, AERIN_REQ: %b, AERIN_ACK: %b, AEROUT_ADDR: %h, AEROUT_REQ: %b, AEROUT_ACK: %b, GOODNESS: %h, ONE_SAMPLE_FINISH: %b, SCHED_FULL: %b", 
//   //          $time, AERIN_ADDR, AERIN_REQ, AERIN_ACK, AEROUT_ADDR, AEROUT_REQ, AEROUT_ACK, GOODNESS, ONE_SAMPLE_FINISH, SCHED_FULL);
//   //end
//   task automatic aer_send (
//     input  logic [AERIN_WIDTH-1:0] addr_in,
//     ref    logic [AERIN_WIDTH-1:0] addr_out,
//     ref    logic          ack,
//     ref    logic          req
// );
//     while (ack) wait_ns(1);
//     addr_out = addr_in;
//     wait_ns(5);
//     req = 1'b1;
//     while (!ack) wait_ns(1);
//     wait_ns(5);
//     req = 1'b0;
// endtask    
// task automatic auto_ack (
//         ref    logic       req,
//         ref    logic       ack,
//         ref    logic [AEROUT_WIDTH-1:0] addr,
//         ref    logic [AEROUT_WIDTH-1:0] neur,
//         ref    logic       verbose
//     );
    
//         forever begin
//             while (~req) wait_ns(1);
//             wait_ns(100);
//             neur = addr;
//             if (verbose)
//                 $display("----- NEURON OUTPUT SPIKE (FROM AER): Event from neuron %d", neur);
//             ack = 1'b1;
//             while (req) wait_ns(1);
//             wait_ns(100);
//             ack = 1'b0;
//         end
// 	endtask

//   	task wait_ns;
//         input   tics_ns;
//         integer tics_ns;
//         #tics_ns;
//     endtask
// endmodule
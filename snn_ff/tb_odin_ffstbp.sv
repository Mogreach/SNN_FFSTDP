`timescale 1ns / 1ps

module ODIN_ffstdp_tb();
  // Parameters
  parameter CLK_PERIOD = 4;

  // Signals
  logic CLK;
  logic RST;
  logic IS_POS;
  logic IS_TRAIN;
  logic SCK;
  logic MOSI;
  logic MISO;
  logic [11:0] AERIN_ADDR;
  logic AERIN_REQ;
  logic [15:0] cnt;
  logic [15:0] pixel_index;
  logic [7:0] aer_neur_spk;
  logic AERIN_ACK;
  logic [9:0] AEROUT_ADDR;
  logic AEROUT_REQ;
  logic AEROUT_ACK;
  logic [31:0] GOODNESS;
  logic ONE_SAMPLE_FINISH;
  logic SCHED_FULL;
  logic auto_ack_verbose;

  // Instantiate the DUT (Device Under Test)
  ODIN_ffstdp #(
    .N(784),
    .M(10)
  ) dut (
    .CLK(CLK),
    .RST(RST),
    .IS_POS(IS_POS),
    .IS_TRAIN(IS_TRAIN),
    .SCK(SCK),
    .MOSI(MOSI),
    .MISO(MISO),
    .AERIN_ADDR(AERIN_ADDR),
    .AERIN_REQ(AERIN_REQ),
    .AERIN_ACK(AERIN_ACK),
    .AEROUT_ADDR(AEROUT_ADDR),
    .AEROUT_REQ(AEROUT_REQ),
    .AEROUT_ACK(AEROUT_ACK),
    .GOODNESS(GOODNESS),
    .ONE_SAMPLE_FINISH(ONE_SAMPLE_FINISH),
    .SCHED_FULL(SCHED_FULL)
  );

  // Clock generation
  initial begin
    CLK = 0;
    forever #(CLK_PERIOD / 2) CLK = ~CLK;
  end
  always @(posedge CLK) begin
    if (cnt == 784)
      cnt <= 0;
    else if(AERIN_REQ && AERIN_ACK)
      cnt <= cnt + 1;
    else
      cnt <= cnt;
  end

    // **读取 TXT 文件**
    parameter int N = 1;   // 样本数
    parameter int T = 8;    // 时间步
    parameter int WIDTH = 784;  // 每个时间步的 bit 数
    
    bit spike_data_reshaped [0:N-1][0:T-1][0:WIDTH-1]; // 存储展开后的数据
    integer file, byte_count;
    bit [7:0] spike_byte;
    integer bit_index = 0;
    integer n_idx = 0, t_idx = 0, w_idx = 0;
  initial begin
      // file = $fopen("D:/BaiduSyncdisk/SNN_FFSTBP/sim/python/simulation_spikes.bin", "rb"); // 以二进制方式读取
      // file = $fopen("D:/BaiduSyncdisk/SNN_FFSTBP/sim/python/all_spikes.bin", "rb"); // 以二进制方式读取
      file = $fopen("D:/OneDrive/SNN-ForwardForward/hardware_sim/all_spikes.bin", "rb");
      
      if (file == 0) begin
          $display("Error: Cannot open file!");
          $finish;
      end

      // 读取所有数据
      while (!$feof(file) && n_idx < N) begin
            byte_count = $fread(spike_byte, file); // 读取 1 字节（8-bit）

            if (byte_count > 0) begin
                for (int i = 7; i >= 0; i--) begin
                    spike_data_reshaped[n_idx][t_idx][w_idx] = spike_byte[i]; // 存入数组
                    // 更新索引
                    w_idx++;
                    if (w_idx == WIDTH) begin
                        w_idx = 0;
                        t_idx++;
                        if (t_idx == T) begin
                            t_idx = 0;
                            n_idx++;
                        end
                    end
                    if (n_idx >= N) break; // 读取到 N 个样本后停止
                end
            end
        end
        $fclose(file);
        $display("Spike data loaded successfully!");
  end




  always @(posedge CLK) begin
    if (ONE_SAMPLE_FINISH)
      IS_POS <= ~IS_POS;
    else
      IS_POS <= IS_POS;
  end

  // Reset and stimulus
  initial begin                 
  auto_ack_verbose = 1'b1;
    fork
      auto_ack(.req(AEROUT_REQ), .ack(AEROUT_ACK), .addr(AEROUT_ADDR), .neur(aer_neur_spk), .verbose(auto_ack_verbose));
    join_none
    // Initialize signals
    RST = 1;
    IS_POS = 0;
    IS_TRAIN = 0;
    SCK = 0;
    MOSI = 0;
    AERIN_ADDR = 12'b0;
    AERIN_REQ = 0;
    AEROUT_ACK = 0;
    cnt = 0;

    // Apply reset
    #20;
    RST = 0;

    // Stimulus
    #20;
    IS_TRAIN = 1;
    IS_POS = 1;
    
    // 重复调用 aer_send 任务
    // for (int sample=0; sample < 5 ; sample++) begin
    //   for (int t =0; t < 16 ; t++) begin
    //     int send_count = 0;
    //     for (int i = 0; i < 784; i++) begin
    //       if (send_count < 256 && $urandom % 2 == 0) begin
    //         aer_send (.addr_in({1'b0, 1'b0, cnt[9:0]}), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
    //         send_count++;
    //       end
    //       wait_ns(10);
    //     end
    //     wait_ns(10);
    //     aer_send (.addr_in({1'b0,1'b1,10'hFF}), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
    //   end
    // end

          // 遍历 N 个样本，每个样本有 T 个时间步
    for (int n = 0; n < N; n++) begin
        int sample_index;
        int time_index;
        int pixel_index;
        for (int t = 0; t < T; t++) begin
            for (int pix = 0; pix < 784; pix++) begin
                sample_index = n;  // 选择当前样本
                time_index = t;    // 选择当前时间步
                pixel_index = pix; // 选择当前像素
                if (spike_data_reshaped[sample_index][time_index][pixel_index] == 1) begin
                  aer_send (.addr_in({1'b0, 1'b0, pixel_index[9:0]}), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
                  wait_ns(10);
                end
            end
            aer_send (.addr_in({1'b0,1'b1,10'hFF}), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
            wait_ns(10);
        end
        
        if (n >= 20-1) begin
            IS_TRAIN = 0;
        end
        $display("GOODNESS: %h", GOODNESS);
    end
    // Check results
    

    // Finish simulation
    #200;
    $finish;
  end

  // Monitor signals
  //initial begin
  //  $monitor("Time: %0t, AERIN_ADDR: %h, AERIN_REQ: %b, AERIN_ACK: %b, AEROUT_ADDR: %h, AEROUT_REQ: %b, AEROUT_ACK: %b, GOODNESS: %h, ONE_SAMPLE_FINISH: %b, SCHED_FULL: %b", 
  //          $time, AERIN_ADDR, AERIN_REQ, AERIN_ACK, AEROUT_ADDR, AEROUT_REQ, AEROUT_ACK, GOODNESS, ONE_SAMPLE_FINISH, SCHED_FULL);
  //end
  task automatic aer_send (
    input  logic [11:0] addr_in,
    ref    logic [11:0] addr_out,
    ref    logic          ack,
    ref    logic          req
);
    while (ack) wait_ns(1);
    addr_out = addr_in;
    wait_ns(5);
    req = 1'b1;
    while (!ack) wait_ns(1);
    wait_ns(5);
    req = 1'b0;
endtask    
task automatic auto_ack (
        ref    logic       req,
        ref    logic       ack,
        ref    logic [11:0] addr,
        ref    logic [11:0] neur,
        ref    logic       verbose
    );
    
        forever begin
            while (~req) wait_ns(1);
            wait_ns(100);
            neur = addr;
            if (verbose)
                $display("----- NEURON OUTPUT SPIKE (FROM AER): Event from neuron %d", neur);
            ack = 1'b1;
            while (req) wait_ns(1);
            wait_ns(100);
            ack = 1'b0;
        end
	endtask

  	task wait_ns;
        input   tics_ns;
        integer tics_ns;
        #tics_ns;
    endtask
endmodule
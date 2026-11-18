`timescale 1ns / 1ps

module tb_cnn_top;
    reg clk;
    reg rst;
    reg start;
    wire done;
    wire [2:0] predicted_class;

    cnn_top uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .predicted_class(predicted_class)
    );

    initial clk = 0;
    always #10 clk = ~clk;

    integer cycle_count;

    initial begin
        rst = 1;
        start = 0;
        cycle_count = 0;

        repeat (5) @(posedge clk);
        rst = 0;
        @(posedge clk);

        start = 1;
        @(posedge clk);
        start = 0;

        while (!done) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        $display("=============================================");
        $display("[QUESTA SIMULATION COMPLETE - RGB CNN]");
        $display("Predicted Class Index : %0d", predicted_class);
        $display("Total Clock Cycles    : %0d cycles", cycle_count);
        $display("Latency @ 50 MHz      : %0.3f ms", (cycle_count * 20.0) / 1_000_000.0);
        $display("=============================================");

        #50;
        $finish;
    end
endmodule

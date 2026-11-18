`timescale 1ns / 1ps

module cnn_top (
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    output reg        done,
    output reg  [2:0] predicted_class
);

    // Memori ROM untuk bobot dan citra input (150x150 grayscale)
    reg signed [7:0] img_mem   [0:22499];
    reg signed [7:0] w_conv1   [0:71];
    reg signed [7:0] w_conv2   [0:1151];
    reg signed [7:0] w_dense   [0:95];

    // Buffer Feature Map (setelah MaxPool)
    // Pool1: 74 x 74 x 8 = 43,808 elemen
    // Pool2: 36 x 36 x 16 = 20,736 elemen
    reg signed [7:0] pool1_mem [0:43807];
    reg signed [7:0] pool2_mem [0:20735];
    reg signed [7:0] gap_vec   [0:15];

    initial begin
        $readmemh("gs/test_images.mem", img_mem);
        $readmemh("gs/conv1_weights.mem", w_conv1);
        $readmemh("gs/conv2_weights.mem", w_conv2);
        $readmemh("gs/dense_weights.mem", w_dense);
    end

    // State Machine
    localparam S_IDLE  = 3'd0,
               S_CONV1 = 3'd1,
               S_CONV2 = 3'd2,
               S_GAP   = 3'd3,
               S_DENSE = 3'd4,
               S_DONE  = 3'd5;

    reg [2:0] state;

    // Variabel indeks loop
    integer oc, ic, py, px, sy, sx, ky, kx;
    integer in_y, in_x, w_idx, in_idx, out_idx;
    reg signed [31:0] acc;
    reg signed [7:0]  val_relu, max_val;
    reg signed [31:0] best_score;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            predicted_class <= 3'd0;
            oc <= 0; py <= 0; px <= 0;
            sy <= 0; sx <= 0; ic <= 0; ky <= 0; kx <= 0;
            acc <= 0; max_val <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc <= 0; py <= 0; px <= 0;
                        sy <= 0; sx <= 0; ky <= 0; kx <= 0;
                        acc <= 0; max_val <= 0;
                        state <= S_CONV1;
                    end
                end

                // Fused Conv1 (3x3 valid, 1->8) + ReLU + MaxPool(2x2) -> 74x74x8
                S_CONV1: begin
                    in_y   = (py * 2) + sy + ky;
                    in_x   = (px * 2) + sx + kx;
                    in_idx = in_y * 150 + in_x;
                    // Urutan Keras: ((ky * 3 + kx) * 1 + 0) * 8 + oc
                    w_idx  = (ky * 3 + kx) * 8 + oc;

                    acc = acc + (img_mem[in_idx] * w_conv1[w_idx]);

                    if (kx < 2) begin
                        kx <= kx + 1;
                    end else begin
                        kx <= 0;
                        if (ky < 2) begin
                            ky <= ky + 1;
                        end else begin
                            ky <= 0;
                            // Requantize & ReLU
                            if (acc <= 0)
                                val_relu = 8'sd0;
                            else if ((acc >>> 7) > 127)
                                val_relu = 8'sd127;
                            else
                                val_relu = acc[14:7];

                            if (val_relu > max_val)
                                max_val = val_relu;

                            acc <= 0;

                            // Lanjut jendela MaxPool 2x2
                            if (sx < 1) begin
                                sx <= sx + 1;
                            end else begin
                                sx <= 0;
                                if (sy < 1) begin
                                    sy <= sy + 1;
                                end else begin
                                    sy <= 0;
                                    // Simpan hasil Pool1 ke RAM
                                    out_idx = (py * 74 + px) * 8 + oc;
                                    pool1_mem[out_idx] <= max_val;
                                    max_val <= 0;

                                    if (px < 73) begin
                                        px <= px + 1;
                                    end else begin
                                        px <= 0;
                                        if (py < 73) begin
                                            py <= py + 1;
                                        end else begin
                                            py <= 0;
                                            if (oc < 7) begin
                                                oc <= oc + 1;
                                            end else begin
                                                // Pindah ke Conv2
                                                oc <= 0; py <= 0; px <= 0;
                                                sy <= 0; sx <= 0; ic <= 0; ky <= 0; kx <= 0;
                                                acc <= 0; max_val <= 0;
                                                state <= S_CONV2;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                // Fused Conv2 (3x3 valid, 8->16) + ReLU + MaxPool(2x2) -> 36x36x16
                S_CONV2: begin
                    in_y   = (py * 2) + sy + ky;
                    in_x   = (px * 2) + sx + kx;
                    in_idx = (in_y * 74 + in_x) * 8 + ic;
                    // Urutan Keras: ((ky * 3 + kx) * 8 + ic) * 16 + oc
                    w_idx  = ((ky * 3 + kx) * 8 + ic) * 16 + oc;

                    acc = acc + (pool1_mem[in_idx] * w_conv2[w_idx]);

                    if (kx < 2) begin
                        kx <= kx + 1;
                    end else begin
                        kx <= 0;
                        if (ky < 2) begin
                            ky <= ky + 1;
                        end else begin
                            ky <= 0;
                            if (ic < 7) begin
                                ic <= ic + 1;
                            end else begin
                                ic <= 0;
                                // Requantize & ReLU
                                if (acc <= 0)
                                    val_relu = 8'sd0;
                                else if ((acc >>> 7) > 127)
                                    val_relu = 8'sd127;
                                else
                                    val_relu = acc[14:7];

                                if (val_relu > max_val)
                                    max_val = val_relu;

                                acc <= 0;

                                if (sx < 1) begin
                                    sx <= sx + 1;
                                end else begin
                                    sx <= 0;
                                    if (sy < 1) begin
                                        sy <= sy + 1;
                                    end else begin
                                        sy <= 0;
                                        out_idx = (py * 36 + px) * 16 + oc;
                                        pool2_mem[out_idx] <= max_val;
                                        max_val <= 0;

                                        if (px < 35) begin
                                            px <= px + 1;
                                        end else begin
                                            px <= 0;
                                            if (py < 35) begin
                                                py <= py + 1;
                                            end else begin
                                                py <= 0;
                                                if (oc < 15) begin
                                                    oc <= oc + 1;
                                                end else begin
                                                    // Pindah ke Global Average Pooling
                                                    oc <= 0; py <= 0; px <= 0; acc <= 0;
                                                    state <= S_GAP;
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                // Global Average Pooling (36x36 = 1296 elemen per channel)
                S_GAP: begin
                    in_idx = (py * 36 + px) * 16 + oc;
                    acc = acc + pool2_mem[in_idx];

                    if (px < 35) begin
                        px <= px + 1;
                    end else begin
                        px <= 0;
                        if (py < 35) begin
                            py <= py + 1;
                        end else begin
                            py <= 0;
                            // Pembagian rata-rata 1296 (approximasi: (acc * 50) >>> 16)
                            gap_vec[oc] <= (acc * 32'sd50) >>> 16;
                            acc <= 0;
                            if (oc < 15) begin
                                oc <= oc + 1;
                            end else begin
                                oc <= 0; ic <= 0; acc <= 0;
                                best_score <= -32'sd2147483647;
                                state <= S_DENSE;
                            end
                        end
                    end
                end

                // Dense Output (16 -> 6) + Argmax
                S_DENSE: begin
                    w_idx = ic * 6 + oc;
                    acc = acc + (gap_vec[ic] * w_dense[w_idx]);

                    if (ic < 15) begin
                        ic <= ic + 1;
                    end else begin
                        ic <= 0;
                        if (acc > best_score) begin
                            best_score <= acc;
                            predicted_class <= oc[2:0];
                        end
                        acc <= 0;
                        if (oc < 5) begin
                            oc <= oc + 1;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start)
                        state <= S_IDLE;
                end
            endcase
        end
    end
endmodule

//weights: [[[0, 0, 2, -3, -1, -4, -1, 0, -5, 1, 3, 1, -1, 1, -1, 2, -3, 2, -4, 2, -3], [34, -23, 4, -19, 11, -19, 41, 56, 19, 30, 35, -1, -30, -36, -73, 63, -89, -36, -50, 60, -12], [11, 68, -17, 16, 5, 33, 7, -30, -25, -25, 27, 58, 13, -52, -27, 1, 39, -93, -17, -22, -9]], [[-7, -45, 28], [6, -21, -22], [5, 95, -52]]]
//intercepts: [[-73, 748, 1077], [-1547, 902, -2131]]
module top (inp, out);
input [83:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight 0 : skip
    //weight 0 : skip
    wire signed [11:0] n_0_0_po_2;
    //weight 2 : 8'sb00000010
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00000010;

    wire signed [11:0] n_0_0_po_3;
    //weight -3 : 8'sb11111101
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11111101;

    wire signed [11:0] n_0_0_po_4;
    //weight -1 : 8'sb11111111
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11111111;

    wire signed [11:0] n_0_0_po_5;
    //weight -4 : 8'sb11111100
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11111100;

    wire signed [11:0] n_0_0_po_6;
    //weight -1 : 8'sb11111111
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb11111111;

    //weight 0 : skip
    wire signed [11:0] n_0_0_po_8;
    //weight -5 : 8'sb11111011
    assign n_0_0_po_8 = $signed({1'b0, inp[35:32]}) * 8'sb11111011;

    wire signed [11:0] n_0_0_po_9;
    //weight 1 : 8'sb00000001
    assign n_0_0_po_9 = $signed({1'b0, inp[39:36]}) * 8'sb00000001;

    wire signed [11:0] n_0_0_po_10;
    //weight 3 : 8'sb00000011
    assign n_0_0_po_10 = $signed({1'b0, inp[43:40]}) * 8'sb00000011;

    wire signed [11:0] n_0_0_po_11;
    //weight 1 : 8'sb00000001
    assign n_0_0_po_11 = $signed({1'b0, inp[47:44]}) * 8'sb00000001;

    wire signed [11:0] n_0_0_po_12;
    //weight -1 : 8'sb11111111
    assign n_0_0_po_12 = $signed({1'b0, inp[51:48]}) * 8'sb11111111;

    wire signed [11:0] n_0_0_po_13;
    //weight 1 : 8'sb00000001
    assign n_0_0_po_13 = $signed({1'b0, inp[55:52]}) * 8'sb00000001;

    wire signed [11:0] n_0_0_po_14;
    //weight -1 : 8'sb11111111
    assign n_0_0_po_14 = $signed({1'b0, inp[59:56]}) * 8'sb11111111;

    wire signed [11:0] n_0_0_po_15;
    //weight 2 : 8'sb00000010
    assign n_0_0_po_15 = $signed({1'b0, inp[63:60]}) * 8'sb00000010;

    wire signed [11:0] n_0_0_po_16;
    //weight -3 : 8'sb11111101
    assign n_0_0_po_16 = $signed({1'b0, inp[67:64]}) * 8'sb11111101;

    wire signed [11:0] n_0_0_po_17;
    //weight 2 : 8'sb00000010
    assign n_0_0_po_17 = $signed({1'b0, inp[71:68]}) * 8'sb00000010;

    wire signed [11:0] n_0_0_po_18;
    //weight -4 : 8'sb11111100
    assign n_0_0_po_18 = $signed({1'b0, inp[75:72]}) * 8'sb11111100;

    wire signed [11:0] n_0_0_po_19;
    //weight 2 : 8'sb00000010
    assign n_0_0_po_19 = $signed({1'b0, inp[79:76]}) * 8'sb00000010;

    wire signed [11:0] n_0_0_po_20;
    //weight -3 : 8'sb11111101
    assign n_0_0_po_20 = $signed({1'b0, inp[83:80]}) * 8'sb11111101;

    wire signed [32:0] n_0_0_sum;
    assign n_0_0_sum = -73 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_5 + n_0_0_po_6 + n_0_0_po_8 + n_0_0_po_9 + n_0_0_po_10 + n_0_0_po_11 + n_0_0_po_12 + n_0_0_po_13 + n_0_0_po_14 + n_0_0_po_15 + n_0_0_po_16 + n_0_0_po_17 + n_0_0_po_18 + n_0_0_po_19 + n_0_0_po_20;
    //relu
    wire [15:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({16{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight 34 : 8'sb00100010
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00100010;

    wire signed [11:0] n_0_1_po_1;
    //weight -23 : 8'sb11101001
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11101001;

    wire signed [11:0] n_0_1_po_2;
    //weight 4 : 8'sb00000100
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00000100;

    wire signed [11:0] n_0_1_po_3;
    //weight -19 : 8'sb11101101
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11101101;

    wire signed [11:0] n_0_1_po_4;
    //weight 11 : 8'sb00001011
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb00001011;

    wire signed [11:0] n_0_1_po_5;
    //weight -19 : 8'sb11101101
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11101101;

    wire signed [11:0] n_0_1_po_6;
    //weight 41 : 8'sb00101001
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb00101001;

    wire signed [11:0] n_0_1_po_7;
    //weight 56 : 8'sb00111000
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00111000;

    wire signed [11:0] n_0_1_po_8;
    //weight 19 : 8'sb00010011
    assign n_0_1_po_8 = $signed({1'b0, inp[35:32]}) * 8'sb00010011;

    wire signed [11:0] n_0_1_po_9;
    //weight 30 : 8'sb00011110
    assign n_0_1_po_9 = $signed({1'b0, inp[39:36]}) * 8'sb00011110;

    wire signed [11:0] n_0_1_po_10;
    //weight 35 : 8'sb00100011
    assign n_0_1_po_10 = $signed({1'b0, inp[43:40]}) * 8'sb00100011;

    wire signed [11:0] n_0_1_po_11;
    //weight -1 : 8'sb11111111
    assign n_0_1_po_11 = $signed({1'b0, inp[47:44]}) * 8'sb11111111;

    wire signed [11:0] n_0_1_po_12;
    //weight -30 : 8'sb11100010
    assign n_0_1_po_12 = $signed({1'b0, inp[51:48]}) * 8'sb11100010;

    wire signed [11:0] n_0_1_po_13;
    //weight -36 : 8'sb11011100
    assign n_0_1_po_13 = $signed({1'b0, inp[55:52]}) * 8'sb11011100;

    wire signed [11:0] n_0_1_po_14;
    //weight -73 : 8'sb10110111
    assign n_0_1_po_14 = $signed({1'b0, inp[59:56]}) * 8'sb10110111;

    wire signed [11:0] n_0_1_po_15;
    //weight 63 : 8'sb00111111
    assign n_0_1_po_15 = $signed({1'b0, inp[63:60]}) * 8'sb00111111;

    wire signed [11:0] n_0_1_po_16;
    //weight -89 : 8'sb10100111
    assign n_0_1_po_16 = $signed({1'b0, inp[67:64]}) * 8'sb10100111;

    wire signed [11:0] n_0_1_po_17;
    //weight -36 : 8'sb11011100
    assign n_0_1_po_17 = $signed({1'b0, inp[71:68]}) * 8'sb11011100;

    wire signed [11:0] n_0_1_po_18;
    //weight -50 : 8'sb11001110
    assign n_0_1_po_18 = $signed({1'b0, inp[75:72]}) * 8'sb11001110;

    wire signed [11:0] n_0_1_po_19;
    //weight 60 : 8'sb00111100
    assign n_0_1_po_19 = $signed({1'b0, inp[79:76]}) * 8'sb00111100;

    wire signed [11:0] n_0_1_po_20;
    //weight -12 : 8'sb11110100
    assign n_0_1_po_20 = $signed({1'b0, inp[83:80]}) * 8'sb11110100;

    wire signed [32:0] n_0_1_sum;
    assign n_0_1_sum = 748 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_5 + n_0_1_po_6 + n_0_1_po_7 + n_0_1_po_8 + n_0_1_po_9 + n_0_1_po_10 + n_0_1_po_11 + n_0_1_po_12 + n_0_1_po_13 + n_0_1_po_14 + n_0_1_po_15 + n_0_1_po_16 + n_0_1_po_17 + n_0_1_po_18 + n_0_1_po_19 + n_0_1_po_20;
    //relu
    wire [15:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({16{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight 11 : 8'sb00001011
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00001011;

    wire signed [11:0] n_0_2_po_1;
    //weight 68 : 8'sb01000100
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb01000100;

    wire signed [11:0] n_0_2_po_2;
    //weight -17 : 8'sb11101111
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11101111;

    wire signed [11:0] n_0_2_po_3;
    //weight 16 : 8'sb00010000
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00010000;

    wire signed [11:0] n_0_2_po_4;
    //weight 5 : 8'sb00000101
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb00000101;

    wire signed [11:0] n_0_2_po_5;
    //weight 33 : 8'sb00100001
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb00100001;

    wire signed [11:0] n_0_2_po_6;
    //weight 7 : 8'sb00000111
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb00000111;

    wire signed [11:0] n_0_2_po_7;
    //weight -30 : 8'sb11100010
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb11100010;

    wire signed [11:0] n_0_2_po_8;
    //weight -25 : 8'sb11100111
    assign n_0_2_po_8 = $signed({1'b0, inp[35:32]}) * 8'sb11100111;

    wire signed [11:0] n_0_2_po_9;
    //weight -25 : 8'sb11100111
    assign n_0_2_po_9 = $signed({1'b0, inp[39:36]}) * 8'sb11100111;

    wire signed [11:0] n_0_2_po_10;
    //weight 27 : 8'sb00011011
    assign n_0_2_po_10 = $signed({1'b0, inp[43:40]}) * 8'sb00011011;

    wire signed [11:0] n_0_2_po_11;
    //weight 58 : 8'sb00111010
    assign n_0_2_po_11 = $signed({1'b0, inp[47:44]}) * 8'sb00111010;

    wire signed [11:0] n_0_2_po_12;
    //weight 13 : 8'sb00001101
    assign n_0_2_po_12 = $signed({1'b0, inp[51:48]}) * 8'sb00001101;

    wire signed [11:0] n_0_2_po_13;
    //weight -52 : 8'sb11001100
    assign n_0_2_po_13 = $signed({1'b0, inp[55:52]}) * 8'sb11001100;

    wire signed [11:0] n_0_2_po_14;
    //weight -27 : 8'sb11100101
    assign n_0_2_po_14 = $signed({1'b0, inp[59:56]}) * 8'sb11100101;

    wire signed [11:0] n_0_2_po_15;
    //weight 1 : 8'sb00000001
    assign n_0_2_po_15 = $signed({1'b0, inp[63:60]}) * 8'sb00000001;

    wire signed [11:0] n_0_2_po_16;
    //weight 39 : 8'sb00100111
    assign n_0_2_po_16 = $signed({1'b0, inp[67:64]}) * 8'sb00100111;

    wire signed [11:0] n_0_2_po_17;
    //weight -93 : 8'sb10100011
    assign n_0_2_po_17 = $signed({1'b0, inp[71:68]}) * 8'sb10100011;

    wire signed [11:0] n_0_2_po_18;
    //weight -17 : 8'sb11101111
    assign n_0_2_po_18 = $signed({1'b0, inp[75:72]}) * 8'sb11101111;

    wire signed [11:0] n_0_2_po_19;
    //weight -22 : 8'sb11101010
    assign n_0_2_po_19 = $signed({1'b0, inp[79:76]}) * 8'sb11101010;

    wire signed [11:0] n_0_2_po_20;
    //weight -9 : 8'sb11110111
    assign n_0_2_po_20 = $signed({1'b0, inp[83:80]}) * 8'sb11110111;

    wire signed [32:0] n_0_2_sum;
    assign n_0_2_sum = 1077 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_5 + n_0_2_po_6 + n_0_2_po_7 + n_0_2_po_8 + n_0_2_po_9 + n_0_2_po_10 + n_0_2_po_11 + n_0_2_po_12 + n_0_2_po_13 + n_0_2_po_14 + n_0_2_po_15 + n_0_2_po_16 + n_0_2_po_17 + n_0_2_po_18 + n_0_2_po_19 + n_0_2_po_20;
    //relu
    wire [15:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({16{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [22:0] n_1_0_po_0;
    //weight -7 : 8'sb11111001
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111001;

    wire signed [22:0] n_1_0_po_1;
    //weight -45 : 8'sb11010011
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb11010011;

    wire signed [22:0] n_1_0_po_2;
    //weight 28 : 8'sb00011100
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb00011100;

    wire signed [25:0] n_1_0_sum;
    assign n_1_0_sum = -1547 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [23:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [22:0] n_1_1_po_0;
    //weight 6 : 8'sb00000110
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000110;

    wire signed [22:0] n_1_1_po_1;
    //weight -21 : 8'sb11101011
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb11101011;

    wire signed [22:0] n_1_1_po_2;
    //weight -22 : 8'sb11101010
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb11101010;

    wire signed [25:0] n_1_1_sum;
    assign n_1_1_sum = 902 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2;
    //relu
    wire [23:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [22:0] n_1_2_po_0;
    //weight 5 : 8'sb00000101
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000101;

    wire signed [22:0] n_1_2_po_1;
    //weight 95 : 8'sb01011111
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb01011111;

    wire signed [22:0] n_1_2_po_2;
    //weight -52 : 8'sb11001100
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb11001100;

    wire signed [25:0] n_1_2_sum;
    assign n_1_2_sum = -2131 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [23:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [24:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [24:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

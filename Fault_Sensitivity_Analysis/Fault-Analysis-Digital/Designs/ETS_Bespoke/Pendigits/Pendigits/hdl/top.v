//weights: [[[16, 32, 18, -41, -44, 2, 0, 10, -12, -23, 36, 0, 6, 37, -13, 31], [-14, 12, -5, 17, 27, -6, -6, -26, -7, 13, 9, -17, -4, 16, 0, 79], [12, 0, 19, -5, -16, 26, -16, 20, 23, -14, 7, 24, 7, -31, -11, -53]], [[7, -4, -19], [-32, -32, 0], [-36, -17, -32], [-45, 10, 16], [9, -73, -7], [-2, 9, 11], [12, -17, -11], [-17, 7, -4], [0, 1, -30], [-2, -5, 9]]]
//intercepts: [[22, -192, -29], [-2150, 1446, 8870, -10964, 551, -11078, -1215, -3588, -2146, -7548]]
module top (inp, out);
input [63:0] inp;
output [3:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight 16 : 8'sb00010000
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00010000;

    wire signed [11:0] n_0_0_po_1;
    //weight 32 : 8'sb00100000
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00100000;

    wire signed [11:0] n_0_0_po_2;
    //weight 18 : 8'sb00010010
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00010010;

    wire signed [11:0] n_0_0_po_3;
    //weight -41 : 8'sb11010111
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11010111;

    wire signed [11:0] n_0_0_po_4;
    //weight -44 : 8'sb11010100
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11010100;

    wire signed [11:0] n_0_0_po_5;
    //weight 2 : 8'sb00000010
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb00000010;

    //weight 0 : skip
    wire signed [11:0] n_0_0_po_7;
    //weight 10 : 8'sb00001010
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00001010;

    wire signed [11:0] n_0_0_po_8;
    //weight -12 : 8'sb11110100
    assign n_0_0_po_8 = $signed({1'b0, inp[35:32]}) * 8'sb11110100;

    wire signed [11:0] n_0_0_po_9;
    //weight -23 : 8'sb11101001
    assign n_0_0_po_9 = $signed({1'b0, inp[39:36]}) * 8'sb11101001;

    wire signed [11:0] n_0_0_po_10;
    //weight 36 : 8'sb00100100
    assign n_0_0_po_10 = $signed({1'b0, inp[43:40]}) * 8'sb00100100;

    //weight 0 : skip
    wire signed [11:0] n_0_0_po_12;
    //weight 6 : 8'sb00000110
    assign n_0_0_po_12 = $signed({1'b0, inp[51:48]}) * 8'sb00000110;

    wire signed [11:0] n_0_0_po_13;
    //weight 37 : 8'sb00100101
    assign n_0_0_po_13 = $signed({1'b0, inp[55:52]}) * 8'sb00100101;

    wire signed [11:0] n_0_0_po_14;
    //weight -13 : 8'sb11110011
    assign n_0_0_po_14 = $signed({1'b0, inp[59:56]}) * 8'sb11110011;

    wire signed [11:0] n_0_0_po_15;
    //weight 31 : 8'sb00011111
    assign n_0_0_po_15 = $signed({1'b0, inp[63:60]}) * 8'sb00011111;

    wire signed [27:0] n_0_0_sum;
    assign n_0_0_sum = 22 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_5 + n_0_0_po_7 + n_0_0_po_8 + n_0_0_po_9 + n_0_0_po_10 + n_0_0_po_12 + n_0_0_po_13 + n_0_0_po_14 + n_0_0_po_15;
    //relu
    wire [15:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({16{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight -14 : 8'sb11110010
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11110010;

    wire signed [11:0] n_0_1_po_1;
    //weight 12 : 8'sb00001100
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00001100;

    wire signed [11:0] n_0_1_po_2;
    //weight -5 : 8'sb11111011
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11111011;

    wire signed [11:0] n_0_1_po_3;
    //weight 17 : 8'sb00010001
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00010001;

    wire signed [11:0] n_0_1_po_4;
    //weight 27 : 8'sb00011011
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb00011011;

    wire signed [11:0] n_0_1_po_5;
    //weight -6 : 8'sb11111010
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11111010;

    wire signed [11:0] n_0_1_po_6;
    //weight -6 : 8'sb11111010
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb11111010;

    wire signed [11:0] n_0_1_po_7;
    //weight -26 : 8'sb11100110
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb11100110;

    wire signed [11:0] n_0_1_po_8;
    //weight -7 : 8'sb11111001
    assign n_0_1_po_8 = $signed({1'b0, inp[35:32]}) * 8'sb11111001;

    wire signed [11:0] n_0_1_po_9;
    //weight 13 : 8'sb00001101
    assign n_0_1_po_9 = $signed({1'b0, inp[39:36]}) * 8'sb00001101;

    wire signed [11:0] n_0_1_po_10;
    //weight 9 : 8'sb00001001
    assign n_0_1_po_10 = $signed({1'b0, inp[43:40]}) * 8'sb00001001;

    wire signed [11:0] n_0_1_po_11;
    //weight -17 : 8'sb11101111
    assign n_0_1_po_11 = $signed({1'b0, inp[47:44]}) * 8'sb11101111;

    wire signed [11:0] n_0_1_po_12;
    //weight -4 : 8'sb11111100
    assign n_0_1_po_12 = $signed({1'b0, inp[51:48]}) * 8'sb11111100;

    wire signed [11:0] n_0_1_po_13;
    //weight 16 : 8'sb00010000
    assign n_0_1_po_13 = $signed({1'b0, inp[55:52]}) * 8'sb00010000;

    //weight 0 : skip
    wire signed [11:0] n_0_1_po_15;
    //weight 79 : 8'sb01001111
    assign n_0_1_po_15 = $signed({1'b0, inp[63:60]}) * 8'sb01001111;

    wire signed [27:0] n_0_1_sum;
    assign n_0_1_sum = -192 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_5 + n_0_1_po_6 + n_0_1_po_7 + n_0_1_po_8 + n_0_1_po_9 + n_0_1_po_10 + n_0_1_po_11 + n_0_1_po_12 + n_0_1_po_13 + n_0_1_po_15;
    //relu
    wire [15:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({16{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight 12 : 8'sb00001100
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00001100;

    //weight 0 : skip
    wire signed [11:0] n_0_2_po_2;
    //weight 19 : 8'sb00010011
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00010011;

    wire signed [11:0] n_0_2_po_3;
    //weight -5 : 8'sb11111011
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11111011;

    wire signed [11:0] n_0_2_po_4;
    //weight -16 : 8'sb11110000
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11110000;

    wire signed [11:0] n_0_2_po_5;
    //weight 26 : 8'sb00011010
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb00011010;

    wire signed [11:0] n_0_2_po_6;
    //weight -16 : 8'sb11110000
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb11110000;

    wire signed [11:0] n_0_2_po_7;
    //weight 20 : 8'sb00010100
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00010100;

    wire signed [11:0] n_0_2_po_8;
    //weight 23 : 8'sb00010111
    assign n_0_2_po_8 = $signed({1'b0, inp[35:32]}) * 8'sb00010111;

    wire signed [11:0] n_0_2_po_9;
    //weight -14 : 8'sb11110010
    assign n_0_2_po_9 = $signed({1'b0, inp[39:36]}) * 8'sb11110010;

    wire signed [11:0] n_0_2_po_10;
    //weight 7 : 8'sb00000111
    assign n_0_2_po_10 = $signed({1'b0, inp[43:40]}) * 8'sb00000111;

    wire signed [11:0] n_0_2_po_11;
    //weight 24 : 8'sb00011000
    assign n_0_2_po_11 = $signed({1'b0, inp[47:44]}) * 8'sb00011000;

    wire signed [11:0] n_0_2_po_12;
    //weight 7 : 8'sb00000111
    assign n_0_2_po_12 = $signed({1'b0, inp[51:48]}) * 8'sb00000111;

    wire signed [11:0] n_0_2_po_13;
    //weight -31 : 8'sb11100001
    assign n_0_2_po_13 = $signed({1'b0, inp[55:52]}) * 8'sb11100001;

    wire signed [11:0] n_0_2_po_14;
    //weight -11 : 8'sb11110101
    assign n_0_2_po_14 = $signed({1'b0, inp[59:56]}) * 8'sb11110101;

    wire signed [11:0] n_0_2_po_15;
    //weight -53 : 8'sb11001011
    assign n_0_2_po_15 = $signed({1'b0, inp[63:60]}) * 8'sb11001011;

    wire signed [27:0] n_0_2_sum;
    assign n_0_2_sum = -29 + n_0_2_po_0 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_5 + n_0_2_po_6 + n_0_2_po_7 + n_0_2_po_8 + n_0_2_po_9 + n_0_2_po_10 + n_0_2_po_11 + n_0_2_po_12 + n_0_2_po_13 + n_0_2_po_14 + n_0_2_po_15;
    //relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [21:0] n_1_0_po_0;
    //weight 7 : 8'sb00000111
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000111;

    wire signed [21:0] n_1_0_po_1;
    //weight -4 : 8'sb11111100
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb11111100;

    wire signed [21:0] n_1_0_po_2;
    //weight -19 : 8'sb11101101
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb11101101;

    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = -2150 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [21:0] n_1_1_po_0;
    //weight -32 : 8'sb11100000
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb11100000;

    wire signed [21:0] n_1_1_po_1;
    //weight -32 : 8'sb11100000
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb11100000;

    //weight 0 : skip
    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = 1446 + n_1_1_po_0 + n_1_1_po_1;
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [21:0] n_1_2_po_0;
    //weight -36 : 8'sb11011100
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb11011100;

    wire signed [21:0] n_1_2_po_1;
    //weight -17 : 8'sb11101111
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb11101111;

    wire signed [21:0] n_1_2_po_2;
    //weight -32 : 8'sb11100000
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb11100000;

    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = 8870 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [22:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_2_sum);

// layer: 1 - neuron: 3
    wire signed [21:0] n_1_3_po_0;
    //weight -45 : 8'sb11010011
    assign n_1_3_po_0 = $signed({1'b0, n_0_0}) * 8'sb11010011;

    wire signed [21:0] n_1_3_po_1;
    //weight 10 : 8'sb00001010
    assign n_1_3_po_1 = $signed({1'b0, n_0_1}) * 8'sb00001010;

    wire signed [21:0] n_1_3_po_2;
    //weight 16 : 8'sb00010000
    assign n_1_3_po_2 = $signed({1'b0, n_0_2}) * 8'sb00010000;

    wire signed [24:0] n_1_3_sum;
    assign n_1_3_sum = -10964 + n_1_3_po_0 + n_1_3_po_1 + n_1_3_po_2;
    //relu
    wire [22:0] n_1_3;
    assign n_1_3 = (n_1_3_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_3_sum);

// layer: 1 - neuron: 4
    wire signed [21:0] n_1_4_po_0;
    //weight 9 : 8'sb00001001
    assign n_1_4_po_0 = $signed({1'b0, n_0_0}) * 8'sb00001001;

    wire signed [21:0] n_1_4_po_1;
    //weight -73 : 8'sb10110111
    assign n_1_4_po_1 = $signed({1'b0, n_0_1}) * 8'sb10110111;

    wire signed [21:0] n_1_4_po_2;
    //weight -7 : 8'sb11111001
    assign n_1_4_po_2 = $signed({1'b0, n_0_2}) * 8'sb11111001;

    wire signed [24:0] n_1_4_sum;
    assign n_1_4_sum = 551 + n_1_4_po_0 + n_1_4_po_1 + n_1_4_po_2;
    //relu
    wire [22:0] n_1_4;
    assign n_1_4 = (n_1_4_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_4_sum);

// layer: 1 - neuron: 5
    wire signed [21:0] n_1_5_po_0;
    //weight -2 : 8'sb11111110
    assign n_1_5_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111110;

    wire signed [21:0] n_1_5_po_1;
    //weight 9 : 8'sb00001001
    assign n_1_5_po_1 = $signed({1'b0, n_0_1}) * 8'sb00001001;

    wire signed [21:0] n_1_5_po_2;
    //weight 11 : 8'sb00001011
    assign n_1_5_po_2 = $signed({1'b0, n_0_2}) * 8'sb00001011;

    wire signed [24:0] n_1_5_sum;
    assign n_1_5_sum = -11078 + n_1_5_po_0 + n_1_5_po_1 + n_1_5_po_2;
    //relu
    wire [22:0] n_1_5;
    assign n_1_5 = (n_1_5_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_5_sum);

// layer: 1 - neuron: 6
    wire signed [21:0] n_1_6_po_0;
    //weight 12 : 8'sb00001100
    assign n_1_6_po_0 = $signed({1'b0, n_0_0}) * 8'sb00001100;

    wire signed [21:0] n_1_6_po_1;
    //weight -17 : 8'sb11101111
    assign n_1_6_po_1 = $signed({1'b0, n_0_1}) * 8'sb11101111;

    wire signed [21:0] n_1_6_po_2;
    //weight -11 : 8'sb11110101
    assign n_1_6_po_2 = $signed({1'b0, n_0_2}) * 8'sb11110101;

    wire signed [24:0] n_1_6_sum;
    assign n_1_6_sum = -1215 + n_1_6_po_0 + n_1_6_po_1 + n_1_6_po_2;
    //relu
    wire [22:0] n_1_6;
    assign n_1_6 = (n_1_6_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_6_sum);

// layer: 1 - neuron: 7
    wire signed [21:0] n_1_7_po_0;
    //weight -17 : 8'sb11101111
    assign n_1_7_po_0 = $signed({1'b0, n_0_0}) * 8'sb11101111;

    wire signed [21:0] n_1_7_po_1;
    //weight 7 : 8'sb00000111
    assign n_1_7_po_1 = $signed({1'b0, n_0_1}) * 8'sb00000111;

    wire signed [21:0] n_1_7_po_2;
    //weight -4 : 8'sb11111100
    assign n_1_7_po_2 = $signed({1'b0, n_0_2}) * 8'sb11111100;

    wire signed [24:0] n_1_7_sum;
    assign n_1_7_sum = -3588 + n_1_7_po_0 + n_1_7_po_1 + n_1_7_po_2;
    //relu
    wire [22:0] n_1_7;
    assign n_1_7 = (n_1_7_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_7_sum);

// layer: 1 - neuron: 8
    //weight 0 : skip
    wire signed [21:0] n_1_8_po_1;
    //weight 1 : 8'sb00000001
    assign n_1_8_po_1 = $signed({1'b0, n_0_1}) * 8'sb00000001;

    wire signed [21:0] n_1_8_po_2;
    //weight -30 : 8'sb11100010
    assign n_1_8_po_2 = $signed({1'b0, n_0_2}) * 8'sb11100010;

    wire signed [24:0] n_1_8_sum;
    assign n_1_8_sum = -2146 + n_1_8_po_1 + n_1_8_po_2;
    //relu
    wire [22:0] n_1_8;
    assign n_1_8 = (n_1_8_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_8_sum);

// layer: 1 - neuron: 9
    wire signed [21:0] n_1_9_po_0;
    //weight -2 : 8'sb11111110
    assign n_1_9_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111110;

    wire signed [21:0] n_1_9_po_1;
    //weight -5 : 8'sb11111011
    assign n_1_9_po_1 = $signed({1'b0, n_0_1}) * 8'sb11111011;

    wire signed [21:0] n_1_9_po_2;
    //weight 9 : 8'sb00001001
    assign n_1_9_po_2 = $signed({1'b0, n_0_2}) * 8'sb00001001;

    wire signed [24:0] n_1_9_sum;
    assign n_1_9_sum = -7548 + n_1_9_po_0 + n_1_9_po_1 + n_1_9_po_2;
    //relu
    wire [22:0] n_1_9;
    assign n_1_9 = (n_1_9_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_9_sum);

// argmax: 10 classes, need 4 bits
// argmax inp: n_1_0, n_1_1, n_1_2, n_1_3, n_1_4, n_1_5, n_1_6, n_1_7, n_1_8, n_1_9
    //comp level 0
    wire cmp_0_0;
    wire [23:0] argmax_val_0_0;
    wire [3:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 4'b0000 : 4'b0001;

    wire cmp_0_2;
    wire [23:0] argmax_val_0_2;
    wire [3:0] argmax_idx_0_2;
    assign {cmp_0_2} = ( n_1_2 >= n_1_3 );
    assign {argmax_val_0_2} = ( cmp_0_2 ) ? n_1_2 : n_1_3;
    assign {argmax_idx_0_2} = ( cmp_0_2 ) ? 4'b0010 : 4'b0011;

    wire cmp_0_4;
    wire [23:0] argmax_val_0_4;
    wire [3:0] argmax_idx_0_4;
    assign {cmp_0_4} = ( n_1_4 >= n_1_5 );
    assign {argmax_val_0_4} = ( cmp_0_4 ) ? n_1_4 : n_1_5;
    assign {argmax_idx_0_4} = ( cmp_0_4 ) ? 4'b0100 : 4'b0101;

    wire cmp_0_6;
    wire [23:0] argmax_val_0_6;
    wire [3:0] argmax_idx_0_6;
    assign {cmp_0_6} = ( n_1_6 >= n_1_7 );
    assign {argmax_val_0_6} = ( cmp_0_6 ) ? n_1_6 : n_1_7;
    assign {argmax_idx_0_6} = ( cmp_0_6 ) ? 4'b0110 : 4'b0111;

    wire cmp_0_8;
    wire [23:0] argmax_val_0_8;
    wire [3:0] argmax_idx_0_8;
    assign {cmp_0_8} = ( n_1_8 >= n_1_9 );
    assign {argmax_val_0_8} = ( cmp_0_8 ) ? n_1_8 : n_1_9;
    assign {argmax_idx_0_8} = ( cmp_0_8 ) ? 4'b1000 : 4'b1001;

    //comp level 1
    wire cmp_1_0;
    wire [23:0] argmax_val_1_0;
    wire [3:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= argmax_val_0_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : argmax_val_0_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : argmax_idx_0_2;

    wire cmp_1_2;
    wire [23:0] argmax_val_1_2;
    wire [3:0] argmax_idx_1_2;
    assign {cmp_1_2} = ( argmax_val_0_4 >= argmax_val_0_6 );
    assign {argmax_val_1_2} = ( cmp_1_2 ) ? argmax_val_0_4 : argmax_val_0_6;
    assign {argmax_idx_1_2} = ( cmp_1_2 ) ? argmax_idx_0_4 : argmax_idx_0_6;

    //comp level 2
    wire cmp_2_0;
    wire [23:0] argmax_val_2_0;
    wire [3:0] argmax_idx_2_0;
    assign {cmp_2_0} = ( argmax_val_1_0 >= argmax_val_1_2 );
    assign {argmax_val_2_0} = ( cmp_2_0 ) ? argmax_val_1_0 : argmax_val_1_2;
    assign {argmax_idx_2_0} = ( cmp_2_0 ) ? argmax_idx_1_0 : argmax_idx_1_2;

    //comp level 3
    wire cmp_3_0;
    wire [23:0] argmax_val_3_0;
    wire [3:0] argmax_idx_3_0;
    assign {cmp_3_0} = ( argmax_val_2_0 >= argmax_val_0_8 );
    assign {argmax_val_3_0} = ( cmp_3_0 ) ? argmax_val_2_0 : argmax_val_0_8;
    assign {argmax_idx_3_0} = ( cmp_3_0 ) ? argmax_idx_2_0 : argmax_idx_0_8;

    assign out = argmax_idx_3_0;

endmodule

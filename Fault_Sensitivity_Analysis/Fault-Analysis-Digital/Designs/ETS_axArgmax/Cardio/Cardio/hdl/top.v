//weights: [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [32, -4, 1, -16, 8, 0, 32, 64, 16, 32, 32, 0, -32, -32, -64, 64, -64, -32, -64, 64, -8], [16, 64, -16, 16, 8, 16, 8, -32, -16, -16, 32, 64, 16, -64, -32, 1, 32, -64, -16, -16, -8]], [[0, -32, 32], [0, -16, -32], [0, 64, -32]]]
//intercepts: [[-8, 64, 64], [-16, 8, -32]]
module top (inp, out);
input [83:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //weight 0 : skip
    //accumulate positive/negative subproducts

    //WARN: only negative weights with relu. Using zero
    wire signed [7:0] n_0_0_sum;
    assign n_0_0_sum = $signed({8{1'b0}});

    wire signed [7:0] n_0_0_sum_f;
    assign n_0_0_sum_f = $signed(n_0_0_sum);
    //relu
    wire [14:0] n_0_0;
    assign n_0_0 = (n_0_0_sum_f<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_0_sum_f);

// layer: 0 - neuron: 1
    wire [3:0] n_0_1_po_0_a;
    assign n_0_1_po_0_a = inp[3:0];
    //mask 15 : [0, 1, 2, 3]
    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_0;
    assign n_0_1_po_0 = (n_0_1_po_0_a[0] + (n_0_1_po_0_a[1] << 1) + (n_0_1_po_0_a[2] << 2) + (n_0_1_po_0_a[3] << 3)) << 5;

    wire [3:0] n_0_1_po_1_a;
    assign n_0_1_po_1_a = inp[7:4];
    //mask 13 : [0, 2, 3]
    //weight abs(-4) : 3'b100
    wire [6:0] n_0_1_po_1;
    assign n_0_1_po_1 = (n_0_1_po_1_a[0] + (n_0_1_po_1_a[2] << 2) + (n_0_1_po_1_a[3] << 3)) << 2;

    wire [3:0] n_0_1_po_2_a;
    assign n_0_1_po_2_a = inp[11:8];
    //mask 13 : [0, 2, 3]
    //weight 1 : 1'b1
    wire [4:0] n_0_1_po_2;
    assign n_0_1_po_2 = (n_0_1_po_2_a[0] + (n_0_1_po_2_a[2] << 2) + (n_0_1_po_2_a[3] << 3));

    wire [3:0] n_0_1_po_3_a;
    assign n_0_1_po_3_a = inp[15:12];
    //mask 15 : [0, 1, 2, 3]
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_3;
    assign n_0_1_po_3 = (n_0_1_po_3_a[0] + (n_0_1_po_3_a[1] << 1) + (n_0_1_po_3_a[2] << 2) + (n_0_1_po_3_a[3] << 3)) << 4;

    wire [3:0] n_0_1_po_4_a;
    assign n_0_1_po_4_a = inp[19:16];
    //mask 14 : [1, 2, 3]
    //weight 8 : 4'b1000
    wire [7:0] n_0_1_po_4;
    assign n_0_1_po_4 = ((n_0_1_po_4_a[1] << 1) + (n_0_1_po_4_a[2] << 2) + (n_0_1_po_4_a[3] << 3)) << 3;

    //weight 0 : skip
    wire [3:0] n_0_1_po_6_a;
    assign n_0_1_po_6_a = inp[27:24];
    //mask 13 : [0, 2, 3]
    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_6;
    assign n_0_1_po_6 = (n_0_1_po_6_a[0] + (n_0_1_po_6_a[2] << 2) + (n_0_1_po_6_a[3] << 3)) << 5;

    wire [3:0] n_0_1_po_7_a;
    assign n_0_1_po_7_a = inp[31:28];
    //mask 14 : [1, 2, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_7;
    assign n_0_1_po_7 = ((n_0_1_po_7_a[1] << 1) + (n_0_1_po_7_a[2] << 2) + (n_0_1_po_7_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_8_a;
    assign n_0_1_po_8_a = inp[35:32];
    //mask 13 : [0, 2, 3]
    //weight 16 : 5'b10000
    wire [8:0] n_0_1_po_8;
    assign n_0_1_po_8 = (n_0_1_po_8_a[0] + (n_0_1_po_8_a[2] << 2) + (n_0_1_po_8_a[3] << 3)) << 4;

    wire [3:0] n_0_1_po_9_a;
    assign n_0_1_po_9_a = inp[39:36];
    //mask 13 : [0, 2, 3]
    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_9;
    assign n_0_1_po_9 = (n_0_1_po_9_a[0] + (n_0_1_po_9_a[2] << 2) + (n_0_1_po_9_a[3] << 3)) << 5;

    wire [3:0] n_0_1_po_10_a;
    assign n_0_1_po_10_a = inp[43:40];
    //mask 14 : [1, 2, 3]
    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_10;
    assign n_0_1_po_10 = ((n_0_1_po_10_a[1] << 1) + (n_0_1_po_10_a[2] << 2) + (n_0_1_po_10_a[3] << 3)) << 5;

    //weight 0 : skip
    wire [3:0] n_0_1_po_12_a;
    assign n_0_1_po_12_a = inp[51:48];
    //mask 14 : [1, 2, 3]
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_12;
    assign n_0_1_po_12 = ((n_0_1_po_12_a[1] << 1) + (n_0_1_po_12_a[2] << 2) + (n_0_1_po_12_a[3] << 3)) << 5;

    wire [3:0] n_0_1_po_13_a;
    assign n_0_1_po_13_a = inp[55:52];
    //mask 13 : [0, 2, 3]
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_13;
    assign n_0_1_po_13 = (n_0_1_po_13_a[0] + (n_0_1_po_13_a[2] << 2) + (n_0_1_po_13_a[3] << 3)) << 5;

    wire [3:0] n_0_1_po_14_a;
    assign n_0_1_po_14_a = inp[59:56];
    //mask 14 : [1, 2, 3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_1_po_14;
    assign n_0_1_po_14 = ((n_0_1_po_14_a[1] << 1) + (n_0_1_po_14_a[2] << 2) + (n_0_1_po_14_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_15_a;
    assign n_0_1_po_15_a = inp[63:60];
    //mask 14 : [1, 2, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_15;
    assign n_0_1_po_15 = ((n_0_1_po_15_a[1] << 1) + (n_0_1_po_15_a[2] << 2) + (n_0_1_po_15_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_16_a;
    assign n_0_1_po_16_a = inp[67:64];
    //mask 14 : [1, 2, 3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_1_po_16;
    assign n_0_1_po_16 = ((n_0_1_po_16_a[1] << 1) + (n_0_1_po_16_a[2] << 2) + (n_0_1_po_16_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_17_a;
    assign n_0_1_po_17_a = inp[71:68];
    //mask 14 : [1, 2, 3]
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_17;
    assign n_0_1_po_17 = ((n_0_1_po_17_a[1] << 1) + (n_0_1_po_17_a[2] << 2) + (n_0_1_po_17_a[3] << 3)) << 5;

    wire [3:0] n_0_1_po_18_a;
    assign n_0_1_po_18_a = inp[75:72];
    //mask 13 : [0, 2, 3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_1_po_18;
    assign n_0_1_po_18 = (n_0_1_po_18_a[0] + (n_0_1_po_18_a[2] << 2) + (n_0_1_po_18_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_19_a;
    assign n_0_1_po_19_a = inp[79:76];
    //mask 14 : [1, 2, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_19;
    assign n_0_1_po_19 = ((n_0_1_po_19_a[1] << 1) + (n_0_1_po_19_a[2] << 2) + (n_0_1_po_19_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_20_a;
    assign n_0_1_po_20_a = inp[83:80];
    //mask 13 : [0, 2, 3]
    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_1_po_20;
    assign n_0_1_po_20 = (n_0_1_po_20_a[0] + (n_0_1_po_20_a[2] << 2) + (n_0_1_po_20_a[3] << 3)) << 3;

    //accumulate positive/negative subproducts
    wire [13:0] n_0_1_sum_pos;
    assign n_0_1_sum_pos = 10'b1000000000 + n_0_1_po_0 + n_0_1_po_2 + n_0_1_po_4 + n_0_1_po_6 + n_0_1_po_7 + n_0_1_po_8 + n_0_1_po_9 + n_0_1_po_10 + n_0_1_po_15 + n_0_1_po_19;
    wire [13:0] n_0_1_sum_neg;
    assign n_0_1_sum_neg = n_0_1_po_1 + n_0_1_po_3 + n_0_1_po_12 + n_0_1_po_13 + n_0_1_po_14 + n_0_1_po_16 + n_0_1_po_17 + n_0_1_po_18 + n_0_1_po_20;
    wire signed [14:0] n_0_1_sum;
    assign n_0_1_sum = $signed({1'b0,n_0_1_sum_pos}) - $signed({1'b0,n_0_1_sum_neg});

    wire signed [14:0] n_0_1_sum_f;
    assign n_0_1_sum_f = $signed(n_0_1_sum);
    //relu
    wire [7:0] n_0_1, n_0_1_qrelu;
    DW01_satrnd #(14, 10, 3) USR_n_0_1 ( .din(n_0_1_sum_f[13:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_1_qrelu));
    assign n_0_1 = (n_0_1_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_1_qrelu);

// layer: 0 - neuron: 2
    wire [3:0] n_0_2_po_0_a;
    assign n_0_2_po_0_a = inp[3:0];
    //mask 15 : [0, 1, 2, 3]
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_0;
    assign n_0_2_po_0 = (n_0_2_po_0_a[0] + (n_0_2_po_0_a[1] << 1) + (n_0_2_po_0_a[2] << 2) + (n_0_2_po_0_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_1_a;
    assign n_0_2_po_1_a = inp[7:4];
    //mask 15 : [0, 1, 2, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_1;
    assign n_0_2_po_1 = (n_0_2_po_1_a[0] + (n_0_2_po_1_a[1] << 1) + (n_0_2_po_1_a[2] << 2) + (n_0_2_po_1_a[3] << 3)) << 6;

    wire [3:0] n_0_2_po_2_a;
    assign n_0_2_po_2_a = inp[11:8];
    //mask 13 : [0, 2, 3]
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_2_po_2;
    assign n_0_2_po_2 = (n_0_2_po_2_a[0] + (n_0_2_po_2_a[2] << 2) + (n_0_2_po_2_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_3_a;
    assign n_0_2_po_3_a = inp[15:12];
    //mask 14 : [1, 2, 3]
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_3;
    assign n_0_2_po_3 = ((n_0_2_po_3_a[1] << 1) + (n_0_2_po_3_a[2] << 2) + (n_0_2_po_3_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_4_a;
    assign n_0_2_po_4_a = inp[19:16];
    //mask 14 : [1, 2, 3]
    //weight 8 : 4'b1000
    wire [7:0] n_0_2_po_4;
    assign n_0_2_po_4 = ((n_0_2_po_4_a[1] << 1) + (n_0_2_po_4_a[2] << 2) + (n_0_2_po_4_a[3] << 3)) << 3;

    wire [3:0] n_0_2_po_5_a;
    assign n_0_2_po_5_a = inp[23:20];
    //mask 13 : [0, 2, 3]
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_5;
    assign n_0_2_po_5 = (n_0_2_po_5_a[0] + (n_0_2_po_5_a[2] << 2) + (n_0_2_po_5_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_6_a;
    assign n_0_2_po_6_a = inp[27:24];
    //mask 15 : [0, 1, 2, 3]
    //weight 8 : 4'b1000
    wire [7:0] n_0_2_po_6;
    assign n_0_2_po_6 = (n_0_2_po_6_a[0] + (n_0_2_po_6_a[1] << 1) + (n_0_2_po_6_a[2] << 2) + (n_0_2_po_6_a[3] << 3)) << 3;

    wire [3:0] n_0_2_po_7_a;
    assign n_0_2_po_7_a = inp[31:28];
    //mask 13 : [0, 2, 3]
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_7;
    assign n_0_2_po_7 = (n_0_2_po_7_a[0] + (n_0_2_po_7_a[2] << 2) + (n_0_2_po_7_a[3] << 3)) << 5;

    wire [3:0] n_0_2_po_8_a;
    assign n_0_2_po_8_a = inp[35:32];
    //mask 15 : [0, 1, 2, 3]
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_2_po_8;
    assign n_0_2_po_8 = (n_0_2_po_8_a[0] + (n_0_2_po_8_a[1] << 1) + (n_0_2_po_8_a[2] << 2) + (n_0_2_po_8_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_9_a;
    assign n_0_2_po_9_a = inp[39:36];
    //mask 13 : [0, 2, 3]
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_2_po_9;
    assign n_0_2_po_9 = (n_0_2_po_9_a[0] + (n_0_2_po_9_a[2] << 2) + (n_0_2_po_9_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_10_a;
    assign n_0_2_po_10_a = inp[43:40];
    //mask 14 : [1, 2, 3]
    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_10;
    assign n_0_2_po_10 = ((n_0_2_po_10_a[1] << 1) + (n_0_2_po_10_a[2] << 2) + (n_0_2_po_10_a[3] << 3)) << 5;

    wire [3:0] n_0_2_po_11_a;
    assign n_0_2_po_11_a = inp[47:44];
    //mask 15 : [0, 1, 2, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_11;
    assign n_0_2_po_11 = (n_0_2_po_11_a[0] + (n_0_2_po_11_a[1] << 1) + (n_0_2_po_11_a[2] << 2) + (n_0_2_po_11_a[3] << 3)) << 6;

    wire [3:0] n_0_2_po_12_a;
    assign n_0_2_po_12_a = inp[51:48];
    //mask 14 : [1, 2, 3]
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_12;
    assign n_0_2_po_12 = ((n_0_2_po_12_a[1] << 1) + (n_0_2_po_12_a[2] << 2) + (n_0_2_po_12_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_13_a;
    assign n_0_2_po_13_a = inp[55:52];
    //mask 13 : [0, 2, 3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_13;
    assign n_0_2_po_13 = (n_0_2_po_13_a[0] + (n_0_2_po_13_a[2] << 2) + (n_0_2_po_13_a[3] << 3)) << 6;

    wire [3:0] n_0_2_po_14_a;
    assign n_0_2_po_14_a = inp[59:56];
    //mask 14 : [1, 2, 3]
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_14;
    assign n_0_2_po_14 = ((n_0_2_po_14_a[1] << 1) + (n_0_2_po_14_a[2] << 2) + (n_0_2_po_14_a[3] << 3)) << 5;

    wire [3:0] n_0_2_po_15_a;
    assign n_0_2_po_15_a = inp[63:60];
    //mask 15 : [0, 1, 2, 3]
    //weight 1 : 1'b1
    wire [4:0] n_0_2_po_15;
    assign n_0_2_po_15 = (n_0_2_po_15_a[0] + (n_0_2_po_15_a[1] << 1) + (n_0_2_po_15_a[2] << 2) + (n_0_2_po_15_a[3] << 3));

    wire [3:0] n_0_2_po_16_a;
    assign n_0_2_po_16_a = inp[67:64];
    //mask 14 : [1, 2, 3]
    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_16;
    assign n_0_2_po_16 = ((n_0_2_po_16_a[1] << 1) + (n_0_2_po_16_a[2] << 2) + (n_0_2_po_16_a[3] << 3)) << 5;

    wire [3:0] n_0_2_po_17_a;
    assign n_0_2_po_17_a = inp[71:68];
    //mask 15 : [0, 1, 2, 3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_17;
    assign n_0_2_po_17 = (n_0_2_po_17_a[0] + (n_0_2_po_17_a[1] << 1) + (n_0_2_po_17_a[2] << 2) + (n_0_2_po_17_a[3] << 3)) << 6;

    wire [3:0] n_0_2_po_18_a;
    assign n_0_2_po_18_a = inp[75:72];
    //mask 14 : [1, 2, 3]
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_2_po_18;
    assign n_0_2_po_18 = ((n_0_2_po_18_a[1] << 1) + (n_0_2_po_18_a[2] << 2) + (n_0_2_po_18_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_19_a;
    assign n_0_2_po_19_a = inp[79:76];
    //mask 14 : [1, 2, 3]
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_2_po_19;
    assign n_0_2_po_19 = ((n_0_2_po_19_a[1] << 1) + (n_0_2_po_19_a[2] << 2) + (n_0_2_po_19_a[3] << 3)) << 4;

    wire [3:0] n_0_2_po_20_a;
    assign n_0_2_po_20_a = inp[83:80];
    //mask 13 : [0, 2, 3]
    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_2_po_20;
    assign n_0_2_po_20 = (n_0_2_po_20_a[0] + (n_0_2_po_20_a[2] << 2) + (n_0_2_po_20_a[3] << 3)) << 3;

    //accumulate positive/negative subproducts
    wire [13:0] n_0_2_sum_pos;
    assign n_0_2_sum_pos = 10'b1000000000 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_5 + n_0_2_po_6 + n_0_2_po_10 + n_0_2_po_11 + n_0_2_po_12 + n_0_2_po_15 + n_0_2_po_16;
    wire [13:0] n_0_2_sum_neg;
    assign n_0_2_sum_neg = n_0_2_po_2 + n_0_2_po_7 + n_0_2_po_8 + n_0_2_po_9 + n_0_2_po_13 + n_0_2_po_14 + n_0_2_po_17 + n_0_2_po_18 + n_0_2_po_19 + n_0_2_po_20;
    wire signed [14:0] n_0_2_sum;
    assign n_0_2_sum = $signed({1'b0,n_0_2_sum_pos}) - $signed({1'b0,n_0_2_sum_neg});

    wire signed [14:0] n_0_2_sum_f;
    assign n_0_2_sum_f = $signed(n_0_2_sum);
    //relu
    wire [7:0] n_0_2, n_0_2_qrelu;
    DW01_satrnd #(14, 10, 3) USR_n_0_2 ( .din(n_0_2_sum_f[13:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_2_qrelu));
    assign n_0_2 = (n_0_2_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_2_qrelu);

// layer: 1 - neuron: 0
    //weight 0 : skip
    wire [7:0] n_1_0_po_1_a;
    assign n_1_0_po_1_a = n_0_1;
    //mask 253 : [0, 2, 3, 4, 5, 6, 7]
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_0_po_1;
    assign n_1_0_po_1 = (n_1_0_po_1_a[0] + (n_1_0_po_1_a[2] << 2) + (n_1_0_po_1_a[3] << 3) + (n_1_0_po_1_a[4] << 4) + (n_1_0_po_1_a[5] << 5) + (n_1_0_po_1_a[6] << 6) + (n_1_0_po_1_a[7] << 7)) << 5;

    wire [7:0] n_1_0_po_2_a;
    assign n_1_0_po_2_a = n_0_2;
    //mask 255 : [0, 1, 2, 3, 4, 5, 6, 7]
    //weight 32 : 6'b100000
    wire [13:0] n_1_0_po_2;
    assign n_1_0_po_2 = (n_1_0_po_2_a[0] + (n_1_0_po_2_a[1] << 1) + (n_1_0_po_2_a[2] << 2) + (n_1_0_po_2_a[3] << 3) + (n_1_0_po_2_a[4] << 4) + (n_1_0_po_2_a[5] << 5) + (n_1_0_po_2_a[6] << 6) + (n_1_0_po_2_a[7] << 7)) << 5;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_0_sum_pos;
    assign n_1_0_sum_pos = n_1_0_po_2;
    wire [14:0] n_1_0_sum_neg;
    assign n_1_0_sum_neg = 8'b10000000 + n_1_0_po_1;
    wire signed [15:0] n_1_0_sum;
    assign n_1_0_sum = $signed({1'b0,n_1_0_sum_pos}) - $signed({1'b0,n_1_0_sum_neg});

    wire signed [15:0] n_1_0_sum_f;
    assign n_1_0_sum_f = $signed(n_1_0_sum);
    //relu
    wire [14:0] n_1_0;
    assign n_1_0 = (n_1_0_sum_f<0) ? $unsigned({15{1'b0}}) : $unsigned(n_1_0_sum_f);

// layer: 1 - neuron: 1
    //weight 0 : skip
    wire [7:0] n_1_1_po_1_a;
    assign n_1_1_po_1_a = n_0_1;
    //mask 253 : [0, 2, 3, 4, 5, 6, 7]
    //weight abs(-16) : 5'b10000
    wire [12:0] n_1_1_po_1;
    assign n_1_1_po_1 = (n_1_1_po_1_a[0] + (n_1_1_po_1_a[2] << 2) + (n_1_1_po_1_a[3] << 3) + (n_1_1_po_1_a[4] << 4) + (n_1_1_po_1_a[5] << 5) + (n_1_1_po_1_a[6] << 6) + (n_1_1_po_1_a[7] << 7)) << 4;

    wire [7:0] n_1_1_po_2_a;
    assign n_1_1_po_2_a = n_0_2;
    //mask 255 : [0, 1, 2, 3, 4, 5, 6, 7]
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_1_po_2;
    assign n_1_1_po_2 = (n_1_1_po_2_a[0] + (n_1_1_po_2_a[1] << 1) + (n_1_1_po_2_a[2] << 2) + (n_1_1_po_2_a[3] << 3) + (n_1_1_po_2_a[4] << 4) + (n_1_1_po_2_a[5] << 5) + (n_1_1_po_2_a[6] << 6) + (n_1_1_po_2_a[7] << 7)) << 5;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_1_sum_pos;
    assign n_1_1_sum_pos = 7'b1000000;
    wire [14:0] n_1_1_sum_neg;
    assign n_1_1_sum_neg = n_1_1_po_1 + n_1_1_po_2;
    wire signed [15:0] n_1_1_sum;
    assign n_1_1_sum = $signed({1'b0,n_1_1_sum_pos}) - $signed({1'b0,n_1_1_sum_neg});

    wire signed [15:0] n_1_1_sum_f;
    assign n_1_1_sum_f = $signed(n_1_1_sum);
    //relu
    wire [14:0] n_1_1;
    assign n_1_1 = (n_1_1_sum_f<0) ? $unsigned({15{1'b0}}) : $unsigned(n_1_1_sum_f);

// layer: 1 - neuron: 2
    //weight 0 : skip
    wire [7:0] n_1_2_po_1_a;
    assign n_1_2_po_1_a = n_0_1;
    //mask 253 : [0, 2, 3, 4, 5, 6, 7]
    //weight 64 : 7'b1000000
    wire [14:0] n_1_2_po_1;
    assign n_1_2_po_1 = (n_1_2_po_1_a[0] + (n_1_2_po_1_a[2] << 2) + (n_1_2_po_1_a[3] << 3) + (n_1_2_po_1_a[4] << 4) + (n_1_2_po_1_a[5] << 5) + (n_1_2_po_1_a[6] << 6) + (n_1_2_po_1_a[7] << 7)) << 6;

    wire [7:0] n_1_2_po_2_a;
    assign n_1_2_po_2_a = n_0_2;
    //mask 254 : [1, 2, 3, 4, 5, 6, 7]
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_2_po_2;
    assign n_1_2_po_2 = ((n_1_2_po_2_a[1] << 1) + (n_1_2_po_2_a[2] << 2) + (n_1_2_po_2_a[3] << 3) + (n_1_2_po_2_a[4] << 4) + (n_1_2_po_2_a[5] << 5) + (n_1_2_po_2_a[6] << 6) + (n_1_2_po_2_a[7] << 7)) << 5;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_2_sum_pos;
    assign n_1_2_sum_pos = n_1_2_po_1;
    wire [14:0] n_1_2_sum_neg;
    assign n_1_2_sum_neg = 9'b100000000 + n_1_2_po_2;
    wire signed [15:0] n_1_2_sum;
    assign n_1_2_sum = $signed({1'b0,n_1_2_sum_pos}) - $signed({1'b0,n_1_2_sum_neg});

    wire signed [15:0] n_1_2_sum_f;
    assign n_1_2_sum_f = $signed(n_1_2_sum);
    //relu
    wire [14:0] n_1_2;
    assign n_1_2 = (n_1_2_sum_f<0) ? $unsigned({15{1'b0}}) : $unsigned(n_1_2_sum_f);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [14:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    //current mask = 000000001000000
    assign {cmp_0_0} = ( {n_1_0[6]} >= {n_1_1[6]} );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    wire cmp_0_2;
    wire [14:0] argmax_val_0_2;
    wire [1:0] argmax_idx_0_2;
    assign {cmp_0_2} = 1;
    assign {argmax_val_0_2} =  n_1_2;
    assign {argmax_idx_0_2} =  2'b10;

    //comp level 1
    wire cmp_1_0;
    wire [14:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    //current mask = 001010000000000
    assign {cmp_1_0} = ( {argmax_val_0_0[12], argmax_val_0_0[10]} >= {argmax_val_0_2[12], argmax_val_0_2[10]} );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : argmax_val_0_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : argmax_idx_0_2;

    assign out = argmax_idx_1_0;

endmodule

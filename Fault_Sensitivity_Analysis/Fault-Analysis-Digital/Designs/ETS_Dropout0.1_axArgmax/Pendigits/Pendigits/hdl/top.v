//weights: [[[32, 64, 32, -64, -64, 1, -2, 16, -32, -64, 64, 1, 8, 64, -32, 64], [-32, 16, -8, 32, 64, -16, -8, -64, -16, 32, 16, -32, -16, 32, -1, 64], [16, -8, 32, -16, -32, 64, -32, 32, 32, -32, 8, 32, 8, -64, -32, -64]], [[16, -4, -32], [-64, -64, 8], [-32, -16, -64], [-64, 16, 32], [16, -64, -16], [-4, 16, 16], [32, -32, -16], [-32, 16, -8], [0, 0, -32], [-2, -8, 16]]]
//intercepts: [[2, -64, -32], [-64, 64, 64, -64, 64, -64, -64, -32, -64, -64]]
module top (inp, out);
input [63:0] inp;
output [3:0] out;

// layer: 0 - neuron: 0
    //weight 32 : 6'b100000
    wire [9:0] n_0_0_po_0;
    assign n_0_0_po_0 = (inp[3:0]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_0_po_1;
    assign n_0_0_po_1 = (inp[7:4]) << 6;

    //weight 32 : 6'b100000
    wire [9:0] n_0_0_po_2;
    assign n_0_0_po_2 = (inp[11:8]) << 5;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_0_po_3;
    assign n_0_0_po_3 = (inp[15:12]) << 6;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_0_po_4;
    assign n_0_0_po_4 = (inp[19:16]) << 6;

    //weight 1 : 1'b1
    wire [4:0] n_0_0_po_5;
    assign n_0_0_po_5 = (inp[23:20]);

    //weight abs(-2) : 2'b10
    wire [5:0] n_0_0_po_6;
    assign n_0_0_po_6 = (inp[27:24]) << 1;

    //weight 16 : 5'b10000
    wire [8:0] n_0_0_po_7;
    assign n_0_0_po_7 = (inp[31:28]) << 4;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_0_po_8;
    assign n_0_0_po_8 = (inp[35:32]) << 5;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_0_po_9;
    assign n_0_0_po_9 = (inp[39:36]) << 6;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_0_po_10;
    assign n_0_0_po_10 = (inp[43:40]) << 6;

    //weight 1 : 1'b1
    wire [4:0] n_0_0_po_11;
    assign n_0_0_po_11 = (inp[47:44]);

    //weight 8 : 4'b1000
    wire [7:0] n_0_0_po_12;
    assign n_0_0_po_12 = (inp[51:48]) << 3;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_0_po_13;
    assign n_0_0_po_13 = (inp[55:52]) << 6;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_0_po_14;
    assign n_0_0_po_14 = (inp[59:56]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_0_po_15;
    assign n_0_0_po_15 = (inp[63:60]) << 6;

    //accumulate positive/negative subproducts
    wire [13:0] n_0_0_sum_pos;
    assign n_0_0_sum_pos = 4'b1000 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_5 + n_0_0_po_7 + n_0_0_po_10 + n_0_0_po_11 + n_0_0_po_12 + n_0_0_po_13 + n_0_0_po_15;
    wire [13:0] n_0_0_sum_neg;
    assign n_0_0_sum_neg = n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_6 + n_0_0_po_8 + n_0_0_po_9 + n_0_0_po_14;
    wire signed [14:0] n_0_0_sum;
    assign n_0_0_sum = $signed({1'b0,n_0_0_sum_pos}) - $signed({1'b0,n_0_0_sum_neg});

    wire signed [14:0] n_0_0_sum_f;
    assign n_0_0_sum_f = $signed(n_0_0_sum);
    //relu
    wire [7:0] n_0_0, n_0_0_qrelu;
    DW01_satrnd #(14, 9, 2) USR_n_0_0 ( .din(n_0_0_sum_f[13:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_0_qrelu));
    assign n_0_0 = (n_0_0_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_0_qrelu);

// layer: 0 - neuron: 1
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_0;
    assign n_0_1_po_0 = (inp[3:0]) << 5;

    //weight 16 : 5'b10000
    wire [8:0] n_0_1_po_1;
    assign n_0_1_po_1 = (inp[7:4]) << 4;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_1_po_2;
    assign n_0_1_po_2 = (inp[11:8]) << 3;

    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_3;
    assign n_0_1_po_3 = (inp[15:12]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_4;
    assign n_0_1_po_4 = (inp[19:16]) << 6;

    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_5;
    assign n_0_1_po_5 = (inp[23:20]) << 4;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_1_po_6;
    assign n_0_1_po_6 = (inp[27:24]) << 3;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_1_po_7;
    assign n_0_1_po_7 = (inp[31:28]) << 6;

    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_8;
    assign n_0_1_po_8 = (inp[35:32]) << 4;

    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_9;
    assign n_0_1_po_9 = (inp[39:36]) << 5;

    //weight 16 : 5'b10000
    wire [8:0] n_0_1_po_10;
    assign n_0_1_po_10 = (inp[43:40]) << 4;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_11;
    assign n_0_1_po_11 = (inp[47:44]) << 5;

    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_12;
    assign n_0_1_po_12 = (inp[51:48]) << 4;

    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_13;
    assign n_0_1_po_13 = (inp[55:52]) << 5;

    //weight abs(-1) : 1'b1
    wire [4:0] n_0_1_po_14;
    assign n_0_1_po_14 = (inp[59:56]);

    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_15;
    assign n_0_1_po_15 = (inp[63:60]) << 6;

    //accumulate positive/negative subproducts
    wire [12:0] n_0_1_sum_pos;
    assign n_0_1_sum_pos = n_0_1_po_1 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_9 + n_0_1_po_10 + n_0_1_po_13 + n_0_1_po_15;
    wire [12:0] n_0_1_sum_neg;
    assign n_0_1_sum_neg = 9'b100000000 + n_0_1_po_0 + n_0_1_po_2 + n_0_1_po_5 + n_0_1_po_6 + n_0_1_po_7 + n_0_1_po_8 + n_0_1_po_11 + n_0_1_po_12 + n_0_1_po_14;
    wire signed [13:0] n_0_1_sum;
    assign n_0_1_sum = $signed({1'b0,n_0_1_sum_pos}) - $signed({1'b0,n_0_1_sum_neg});

    wire signed [13:0] n_0_1_sum_f;
    assign n_0_1_sum_f = $signed(n_0_1_sum);
    //relu
    wire [7:0] n_0_1, n_0_1_qrelu;
    DW01_satrnd #(13, 9, 2) USR_n_0_1 ( .din(n_0_1_sum_f[12:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_1_qrelu));
    assign n_0_1 = (n_0_1_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_1_qrelu);

// layer: 0 - neuron: 2
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_0;
    assign n_0_2_po_0 = (inp[3:0]) << 4;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_2_po_1;
    assign n_0_2_po_1 = (inp[7:4]) << 3;

    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_2;
    assign n_0_2_po_2 = (inp[11:8]) << 5;

    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_2_po_3;
    assign n_0_2_po_3 = (inp[15:12]) << 4;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_4;
    assign n_0_2_po_4 = (inp[19:16]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_5;
    assign n_0_2_po_5 = (inp[23:20]) << 6;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_6;
    assign n_0_2_po_6 = (inp[27:24]) << 5;

    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_7;
    assign n_0_2_po_7 = (inp[31:28]) << 5;

    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_8;
    assign n_0_2_po_8 = (inp[35:32]) << 5;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_9;
    assign n_0_2_po_9 = (inp[39:36]) << 5;

    //weight 8 : 4'b1000
    wire [7:0] n_0_2_po_10;
    assign n_0_2_po_10 = (inp[43:40]) << 3;

    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_11;
    assign n_0_2_po_11 = (inp[47:44]) << 5;

    //weight 8 : 4'b1000
    wire [7:0] n_0_2_po_12;
    assign n_0_2_po_12 = (inp[51:48]) << 3;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_13;
    assign n_0_2_po_13 = (inp[55:52]) << 6;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_14;
    assign n_0_2_po_14 = (inp[59:56]) << 5;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_15;
    assign n_0_2_po_15 = (inp[63:60]) << 6;

    //accumulate positive/negative subproducts
    wire [13:0] n_0_2_sum_pos;
    assign n_0_2_sum_pos = n_0_2_po_0 + n_0_2_po_2 + n_0_2_po_5 + n_0_2_po_7 + n_0_2_po_8 + n_0_2_po_10 + n_0_2_po_11 + n_0_2_po_12;
    wire [13:0] n_0_2_sum_neg;
    assign n_0_2_sum_neg = 8'b10000000 + n_0_2_po_1 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_6 + n_0_2_po_9 + n_0_2_po_13 + n_0_2_po_14 + n_0_2_po_15;
    wire signed [14:0] n_0_2_sum;
    assign n_0_2_sum = $signed({1'b0,n_0_2_sum_pos}) - $signed({1'b0,n_0_2_sum_neg});

    wire signed [14:0] n_0_2_sum_f;
    assign n_0_2_sum_f = $signed(n_0_2_sum);
    //relu
    wire [7:0] n_0_2, n_0_2_qrelu;
    DW01_satrnd #(14, 9, 2) USR_n_0_2 ( .din(n_0_2_sum_f[13:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_2_qrelu));
    assign n_0_2 = (n_0_2_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_2_qrelu);

// layer: 1 - neuron: 0
    //weight 16 : 5'b10000
    wire [12:0] n_1_0_po_0;
    assign n_1_0_po_0 = (n_0_0) << 4;

    //weight abs(-4) : 3'b100
    wire [10:0] n_1_0_po_1;
    assign n_1_0_po_1 = (n_0_1) << 2;

    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_0_po_2;
    assign n_1_0_po_2 = (n_0_2) << 5;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_0_sum_pos;
    assign n_1_0_sum_pos = n_1_0_po_0;
    wire [14:0] n_1_0_sum_neg;
    assign n_1_0_sum_neg = 11'b10000000000 + n_1_0_po_1 + n_1_0_po_2;
    wire signed [15:0] n_1_0_sum;
    assign n_1_0_sum = $signed({1'b0,n_1_0_sum_pos}) - $signed({1'b0,n_1_0_sum_neg});

    wire signed [15:0] n_1_0_sum_f;
    assign n_1_0_sum_f = $signed(n_1_0_sum);
    //relu
    wire [15:0] n_1_0;
    assign n_1_0 = (n_1_0_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_0_sum_f);

// layer: 1 - neuron: 1
    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_1_po_0;
    assign n_1_1_po_0 = (n_0_0) << 6;

    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_1_po_1;
    assign n_1_1_po_1 = (n_0_1) << 6;

    //weight 8 : 4'b1000
    wire [11:0] n_1_1_po_2;
    assign n_1_1_po_2 = (n_0_2) << 3;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_1_sum_pos;
    assign n_1_1_sum_pos = 11'b10000000000 + n_1_1_po_2;
    wire [15:0] n_1_1_sum_neg;
    assign n_1_1_sum_neg = n_1_1_po_0 + n_1_1_po_1;
    wire signed [16:0] n_1_1_sum;
    assign n_1_1_sum = $signed({1'b0,n_1_1_sum_pos}) - $signed({1'b0,n_1_1_sum_neg});

    wire signed [16:0] n_1_1_sum_f;
    assign n_1_1_sum_f = $signed(n_1_1_sum);
    //relu
    wire [15:0] n_1_1;
    assign n_1_1 = (n_1_1_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_1_sum_f);

// layer: 1 - neuron: 2
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_2_po_0;
    assign n_1_2_po_0 = (n_0_0) << 5;

    //weight abs(-16) : 5'b10000
    wire [12:0] n_1_2_po_1;
    assign n_1_2_po_1 = (n_0_1) << 4;

    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_2_po_2;
    assign n_1_2_po_2 = (n_0_2) << 6;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_2_sum_pos;
    assign n_1_2_sum_pos = 11'b10000000000;
    wire [15:0] n_1_2_sum_neg;
    assign n_1_2_sum_neg = n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    wire signed [16:0] n_1_2_sum;
    assign n_1_2_sum = $signed({1'b0,n_1_2_sum_pos}) - $signed({1'b0,n_1_2_sum_neg});

    wire signed [16:0] n_1_2_sum_f;
    assign n_1_2_sum_f = $signed(n_1_2_sum);
    //relu
    wire [15:0] n_1_2;
    assign n_1_2 = (n_1_2_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_2_sum_f);

// layer: 1 - neuron: 3
    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_3_po_0;
    assign n_1_3_po_0 = (n_0_0) << 6;

    //weight 16 : 5'b10000
    wire [12:0] n_1_3_po_1;
    assign n_1_3_po_1 = (n_0_1) << 4;

    //weight 32 : 6'b100000
    wire [13:0] n_1_3_po_2;
    assign n_1_3_po_2 = (n_0_2) << 5;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_3_sum_pos;
    assign n_1_3_sum_pos = n_1_3_po_1 + n_1_3_po_2;
    wire [15:0] n_1_3_sum_neg;
    assign n_1_3_sum_neg = 11'b10000000000 + n_1_3_po_0;
    wire signed [16:0] n_1_3_sum;
    assign n_1_3_sum = $signed({1'b0,n_1_3_sum_pos}) - $signed({1'b0,n_1_3_sum_neg});

    wire signed [16:0] n_1_3_sum_f;
    assign n_1_3_sum_f = $signed(n_1_3_sum);
    //relu
    wire [15:0] n_1_3;
    assign n_1_3 = (n_1_3_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_3_sum_f);

// layer: 1 - neuron: 4
    //weight 16 : 5'b10000
    wire [12:0] n_1_4_po_0;
    assign n_1_4_po_0 = (n_0_0) << 4;

    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_4_po_1;
    assign n_1_4_po_1 = (n_0_1) << 6;

    //weight abs(-16) : 5'b10000
    wire [12:0] n_1_4_po_2;
    assign n_1_4_po_2 = (n_0_2) << 4;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_4_sum_pos;
    assign n_1_4_sum_pos = 11'b10000000000 + n_1_4_po_0;
    wire [15:0] n_1_4_sum_neg;
    assign n_1_4_sum_neg = n_1_4_po_1 + n_1_4_po_2;
    wire signed [16:0] n_1_4_sum;
    assign n_1_4_sum = $signed({1'b0,n_1_4_sum_pos}) - $signed({1'b0,n_1_4_sum_neg});

    wire signed [16:0] n_1_4_sum_f;
    assign n_1_4_sum_f = $signed(n_1_4_sum);
    //relu
    wire [15:0] n_1_4;
    assign n_1_4 = (n_1_4_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_4_sum_f);

// layer: 1 - neuron: 5
    //weight abs(-4) : 3'b100
    wire [10:0] n_1_5_po_0;
    assign n_1_5_po_0 = (n_0_0) << 2;

    //weight 16 : 5'b10000
    wire [12:0] n_1_5_po_1;
    assign n_1_5_po_1 = (n_0_1) << 4;

    //weight 16 : 5'b10000
    wire [12:0] n_1_5_po_2;
    assign n_1_5_po_2 = (n_0_2) << 4;

    //accumulate positive/negative subproducts
    wire [13:0] n_1_5_sum_pos;
    assign n_1_5_sum_pos = n_1_5_po_1 + n_1_5_po_2;
    wire [13:0] n_1_5_sum_neg;
    assign n_1_5_sum_neg = 11'b10000000000 + n_1_5_po_0;
    wire signed [14:0] n_1_5_sum;
    assign n_1_5_sum = $signed({1'b0,n_1_5_sum_pos}) - $signed({1'b0,n_1_5_sum_neg});

    wire signed [14:0] n_1_5_sum_f;
    assign n_1_5_sum_f = $signed(n_1_5_sum);
    //relu
    wire [15:0] n_1_5;
    assign n_1_5 = (n_1_5_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_5_sum_f);

// layer: 1 - neuron: 6
    //weight 32 : 6'b100000
    wire [13:0] n_1_6_po_0;
    assign n_1_6_po_0 = (n_0_0) << 5;

    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_6_po_1;
    assign n_1_6_po_1 = (n_0_1) << 5;

    //weight abs(-16) : 5'b10000
    wire [12:0] n_1_6_po_2;
    assign n_1_6_po_2 = (n_0_2) << 4;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_6_sum_pos;
    assign n_1_6_sum_pos = n_1_6_po_0;
    wire [14:0] n_1_6_sum_neg;
    assign n_1_6_sum_neg = 11'b10000000000 + n_1_6_po_1 + n_1_6_po_2;
    wire signed [15:0] n_1_6_sum;
    assign n_1_6_sum = $signed({1'b0,n_1_6_sum_pos}) - $signed({1'b0,n_1_6_sum_neg});

    wire signed [15:0] n_1_6_sum_f;
    assign n_1_6_sum_f = $signed(n_1_6_sum);
    //relu
    wire [15:0] n_1_6;
    assign n_1_6 = (n_1_6_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_6_sum_f);

// layer: 1 - neuron: 7
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_7_po_0;
    assign n_1_7_po_0 = (n_0_0) << 5;

    //weight 16 : 5'b10000
    wire [12:0] n_1_7_po_1;
    assign n_1_7_po_1 = (n_0_1) << 4;

    //weight abs(-8) : 4'b1000
    wire [11:0] n_1_7_po_2;
    assign n_1_7_po_2 = (n_0_2) << 3;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_7_sum_pos;
    assign n_1_7_sum_pos = n_1_7_po_1;
    wire [14:0] n_1_7_sum_neg;
    assign n_1_7_sum_neg = 10'b1000000000 + n_1_7_po_0 + n_1_7_po_2;
    wire signed [15:0] n_1_7_sum;
    assign n_1_7_sum = $signed({1'b0,n_1_7_sum_pos}) - $signed({1'b0,n_1_7_sum_neg});

    wire signed [15:0] n_1_7_sum_f;
    assign n_1_7_sum_f = $signed(n_1_7_sum);
    //relu
    wire [15:0] n_1_7;
    assign n_1_7 = (n_1_7_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_7_sum_f);

// layer: 1 - neuron: 8
    //weight 0 : skip
    //weight 0 : skip
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_8_po_2;
    assign n_1_8_po_2 = (n_0_2) << 5;

    //accumulate positive/negative subproducts

    //WARN: only negative weights with relu. Using zero
    wire signed [15:0] n_1_8_sum;
    assign n_1_8_sum = $signed({16{1'b0}});

    wire signed [15:0] n_1_8_sum_f;
    assign n_1_8_sum_f = $signed(n_1_8_sum);
    //relu
    wire [15:0] n_1_8;
    assign n_1_8 = (n_1_8_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_8_sum_f);

// layer: 1 - neuron: 9
    //weight abs(-2) : 2'b10
    wire [9:0] n_1_9_po_0;
    assign n_1_9_po_0 = (n_0_0) << 1;

    //weight abs(-8) : 4'b1000
    wire [11:0] n_1_9_po_1;
    assign n_1_9_po_1 = (n_0_1) << 3;

    //weight 16 : 5'b10000
    wire [12:0] n_1_9_po_2;
    assign n_1_9_po_2 = (n_0_2) << 4;

    //accumulate positive/negative subproducts
    wire [12:0] n_1_9_sum_pos;
    assign n_1_9_sum_pos = n_1_9_po_2;
    wire [12:0] n_1_9_sum_neg;
    assign n_1_9_sum_neg = 11'b10000000000 + n_1_9_po_0 + n_1_9_po_1;
    wire signed [13:0] n_1_9_sum;
    assign n_1_9_sum = $signed({1'b0,n_1_9_sum_pos}) - $signed({1'b0,n_1_9_sum_neg});

    wire signed [13:0] n_1_9_sum_f;
    assign n_1_9_sum_f = $signed(n_1_9_sum);
    //relu
    wire [15:0] n_1_9;
    assign n_1_9 = (n_1_9_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_9_sum_f);

// argmax: 10 classes, need 4 bits
// argmax inp: n_1_0, n_1_1, n_1_2, n_1_3, n_1_4, n_1_5, n_1_6, n_1_7, n_1_8, n_1_9
    //comp level 0
    wire cmp_0_0;
    wire [15:0] argmax_val_0_0;
    wire [3:0] argmax_idx_0_0;
    //current mask = 0001000000000000
    assign {cmp_0_0} = ( {n_1_0[12]} >= {n_1_3[12]} );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_3;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 4'b0000 : 4'b0011;

    wire cmp_0_2;
    wire [15:0] argmax_val_0_2;
    wire [3:0] argmax_idx_0_2;
    //current mask = 0000010000000000
    assign {cmp_0_2} = ( {n_1_1[10]} >= {n_1_5[10]} );
    assign {argmax_val_0_2} = ( cmp_0_2 ) ? n_1_1 : n_1_5;
    assign {argmax_idx_0_2} = ( cmp_0_2 ) ? 4'b0001 : 4'b0101;

    wire cmp_0_4;
    wire [15:0] argmax_val_0_4;
    wire [3:0] argmax_idx_0_4;
    //current mask = 0001000000000000
    assign {cmp_0_4} = ( {n_1_2[12]} >= {n_1_6[12]} );
    assign {argmax_val_0_4} = ( cmp_0_4 ) ? n_1_2 : n_1_6;
    assign {argmax_idx_0_4} = ( cmp_0_4 ) ? 4'b0010 : 4'b0110;

    wire cmp_0_6;
    wire [15:0] argmax_val_0_6;
    wire [3:0] argmax_idx_0_6;
    //current mask = 0000100000000000
    assign {cmp_0_6} = ( {n_1_4[11]} >= {n_1_7[11]} );
    assign {argmax_val_0_6} = ( cmp_0_6 ) ? n_1_4 : n_1_7;
    assign {argmax_idx_0_6} = ( cmp_0_6 ) ? 4'b0100 : 4'b0111;

    wire cmp_0_8;
    wire [15:0] argmax_val_0_8;
    wire [3:0] argmax_idx_0_8;
    //current mask = 1000000000000000
    assign {cmp_0_8} = ( {n_1_9[15]} >= {n_1_8[15]} );
    assign {argmax_val_0_8} = ( cmp_0_8 ) ? n_1_9 : n_1_8;
    assign {argmax_idx_0_8} = ( cmp_0_8 ) ? 4'b1001 : 4'b1000;

    //comp level 1
    wire cmp_1_0;
    wire [15:0] argmax_val_1_0;
    wire [3:0] argmax_idx_1_0;
    //current mask = 0000100000000000
    assign {cmp_1_0} = ( {argmax_val_0_0[11]} >= {argmax_val_0_6[11]} );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : argmax_val_0_6;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : argmax_idx_0_6;

    wire cmp_1_2;
    wire [15:0] argmax_val_1_2;
    wire [3:0] argmax_idx_1_2;
    //current mask = 0000100000000000
    assign {cmp_1_2} = ( {argmax_val_0_2[11]} >= {argmax_val_0_8[11]} );
    assign {argmax_val_1_2} = ( cmp_1_2 ) ? argmax_val_0_2 : argmax_val_0_8;
    assign {argmax_idx_1_2} = ( cmp_1_2 ) ? argmax_idx_0_2 : argmax_idx_0_8;

    wire cmp_1_4;
    wire [15:0] argmax_val_1_4;
    wire [3:0] argmax_idx_1_4;
    assign {cmp_1_4} = 1;
    assign {argmax_val_1_4} =  argmax_val_0_4;
    assign {argmax_idx_1_4} =  argmax_idx_0_4;

    //comp level 2
    wire cmp_2_0;
    wire [15:0] argmax_val_2_0;
    wire [3:0] argmax_idx_2_0;
    //current mask = 0000011000000000
    assign {cmp_2_0} = ( {argmax_val_1_4[10], argmax_val_1_4[9]} >= {argmax_val_1_2[10], argmax_val_1_2[9]} );
    assign {argmax_val_2_0} = ( cmp_2_0 ) ? argmax_val_1_4 : argmax_val_1_2;
    assign {argmax_idx_2_0} = ( cmp_2_0 ) ? argmax_idx_1_4 : argmax_idx_1_2;

    wire cmp_2_2;
    wire [15:0] argmax_val_2_2;
    wire [3:0] argmax_idx_2_2;
    assign {cmp_2_2} = 1;
    assign {argmax_val_2_2} =  argmax_val_1_0;
    assign {argmax_idx_2_2} =  argmax_idx_1_0;

    //comp level 3
    wire cmp_3_0;
    wire [15:0] argmax_val_3_0;
    wire [3:0] argmax_idx_3_0;
    //current mask = 0001100101000000
    assign {cmp_3_0} = ( {argmax_val_2_0[12], argmax_val_2_0[11], argmax_val_2_0[8], argmax_val_2_0[6]} >= {argmax_val_2_2[12], argmax_val_2_2[11], argmax_val_2_2[8], argmax_val_2_2[6]} );
    assign {argmax_val_3_0} = ( cmp_3_0 ) ? argmax_val_2_0 : argmax_val_2_2;
    assign {argmax_idx_3_0} = ( cmp_3_0 ) ? argmax_idx_2_0 : argmax_idx_2_2;

    assign out = argmax_idx_3_0;

endmodule

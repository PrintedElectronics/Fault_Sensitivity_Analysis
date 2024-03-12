//weights: [[[-4, -4, -8, -2, -8, -4, -4], [16, 4, 16, 64, -32, -16, -64], [64, 64, -32, 64, 64, -32, 64]], [[0, 64, -16], [0, -64, 32], [0, -64, -32]]]
//intercepts: [[-8, 4, -64], [-64, -64, 64]]
module top (inp, out);
input [27:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight abs(-4) : 3'b100
    wire [6:0] n_0_0_po_0;
    assign n_0_0_po_0 = (inp[3:0]) << 2;

    //weight abs(-4) : 3'b100
    wire [6:0] n_0_0_po_1;
    assign n_0_0_po_1 = (inp[7:4]) << 2;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_0_po_2;
    assign n_0_0_po_2 = (inp[11:8]) << 3;

    //weight abs(-2) : 2'b10
    wire [5:0] n_0_0_po_3;
    assign n_0_0_po_3 = (inp[15:12]) << 1;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_0_po_4;
    assign n_0_0_po_4 = (inp[19:16]) << 3;

    //weight abs(-4) : 3'b100
    wire [6:0] n_0_0_po_5;
    assign n_0_0_po_5 = (inp[23:20]) << 2;

    //weight abs(-4) : 3'b100
    wire [6:0] n_0_0_po_6;
    assign n_0_0_po_6 = (inp[27:24]) << 2;

    //accumulate positive/negative subproducts

    //WARN: only negative weights with relu. Using zero
    wire signed [11:0] n_0_0_sum;
    assign n_0_0_sum = $signed({12{1'b0}});

    wire signed [11:0] n_0_0_sum_f;
    assign n_0_0_sum_f = $signed(n_0_0_sum);
    //relu
    wire [7:0] n_0_0, n_0_0_qrelu;
    DW01_satrnd #(11, 7, 0) USR_n_0_0 ( .din(n_0_0_sum_f[10:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_0_qrelu));
    assign n_0_0 = (n_0_0_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_0_qrelu);

// layer: 0 - neuron: 1
    //weight 16 : 5'b10000
    wire [8:0] n_0_1_po_0;
    assign n_0_1_po_0 = (inp[3:0]) << 4;

    //weight 4 : 3'b100
    wire [6:0] n_0_1_po_1;
    assign n_0_1_po_1 = (inp[7:4]) << 2;

    //weight 16 : 5'b10000
    wire [8:0] n_0_1_po_2;
    assign n_0_1_po_2 = (inp[11:8]) << 4;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_3;
    assign n_0_1_po_3 = (inp[15:12]) << 6;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_4;
    assign n_0_1_po_4 = (inp[19:16]) << 5;

    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_5;
    assign n_0_1_po_5 = (inp[23:20]) << 4;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_1_po_6;
    assign n_0_1_po_6 = (inp[27:24]) << 6;

    //accumulate positive/negative subproducts
    wire [11:0] n_0_1_sum_pos;
    assign n_0_1_sum_pos = 7'b1000000 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3;
    wire [11:0] n_0_1_sum_neg;
    assign n_0_1_sum_neg = n_0_1_po_4 + n_0_1_po_5 + n_0_1_po_6;
    wire signed [12:0] n_0_1_sum;
    assign n_0_1_sum = $signed({1'b0,n_0_1_sum_pos}) - $signed({1'b0,n_0_1_sum_neg});

    wire signed [12:0] n_0_1_sum_f;
    assign n_0_1_sum_f = $signed(n_0_1_sum);
    //relu
    wire [7:0] n_0_1, n_0_1_qrelu;
    DW01_satrnd #(12, 7, 0) USR_n_0_1 ( .din(n_0_1_sum_f[11:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_1_qrelu));
    assign n_0_1 = (n_0_1_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_1_qrelu);

// layer: 0 - neuron: 2
    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_0;
    assign n_0_2_po_0 = (inp[3:0]) << 6;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_1;
    assign n_0_2_po_1 = (inp[7:4]) << 6;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_2;
    assign n_0_2_po_2 = (inp[11:8]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_3;
    assign n_0_2_po_3 = (inp[15:12]) << 6;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_4;
    assign n_0_2_po_4 = (inp[19:16]) << 6;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_5;
    assign n_0_2_po_5 = (inp[23:20]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_6;
    assign n_0_2_po_6 = (inp[27:24]) << 6;

    //accumulate positive/negative subproducts
    wire [13:0] n_0_2_sum_pos;
    assign n_0_2_sum_pos = n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_6;
    wire [13:0] n_0_2_sum_neg;
    assign n_0_2_sum_neg = 11'b10000000000 + n_0_2_po_2 + n_0_2_po_5;
    wire signed [14:0] n_0_2_sum;
    assign n_0_2_sum = $signed({1'b0,n_0_2_sum_pos}) - $signed({1'b0,n_0_2_sum_neg});

    wire signed [14:0] n_0_2_sum_f;
    assign n_0_2_sum_f = $signed(n_0_2_sum);
    //relu
    wire [7:0] n_0_2, n_0_2_qrelu;
    DW01_satrnd #(14, 7, 0) USR_n_0_2 ( .din(n_0_2_sum_f[13:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_2_qrelu));
    assign n_0_2 = (n_0_2_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_2_qrelu);

// layer: 1 - neuron: 0
    //weight 0 : skip
    //weight 64 : 7'b1000000
    wire [14:0] n_1_0_po_1;
    assign n_1_0_po_1 = (n_0_1) << 6;

    //weight abs(-16) : 5'b10000
    wire [12:0] n_1_0_po_2;
    assign n_1_0_po_2 = (n_0_2) << 4;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_0_sum_pos;
    assign n_1_0_sum_pos = n_1_0_po_1;
    wire [14:0] n_1_0_sum_neg;
    assign n_1_0_sum_neg = 13'b1000000000000 + n_1_0_po_2;
    wire signed [15:0] n_1_0_sum;
    assign n_1_0_sum = $signed({1'b0,n_1_0_sum_pos}) - $signed({1'b0,n_1_0_sum_neg});

    wire signed [15:0] n_1_0_sum_f;
    assign n_1_0_sum_f = $signed(n_1_0_sum);
    //relu
    wire [15:0] n_1_0;
    assign n_1_0 = (n_1_0_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_0_sum_f);

// layer: 1 - neuron: 1
    //weight 0 : skip
    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_1_po_1;
    assign n_1_1_po_1 = (n_0_1) << 6;

    //weight 32 : 6'b100000
    wire [13:0] n_1_1_po_2;
    assign n_1_1_po_2 = (n_0_2) << 5;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_1_sum_pos;
    assign n_1_1_sum_pos = n_1_1_po_2;
    wire [15:0] n_1_1_sum_neg;
    assign n_1_1_sum_neg = 13'b1000000000000 + n_1_1_po_1;
    wire signed [16:0] n_1_1_sum;
    assign n_1_1_sum = $signed({1'b0,n_1_1_sum_pos}) - $signed({1'b0,n_1_1_sum_neg});

    wire signed [16:0] n_1_1_sum_f;
    assign n_1_1_sum_f = $signed(n_1_1_sum);
    //relu
    wire [15:0] n_1_1;
    assign n_1_1 = (n_1_1_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_1_sum_f);

// layer: 1 - neuron: 2
    //weight 0 : skip
    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_2_po_1;
    assign n_1_2_po_1 = (n_0_1) << 6;

    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_2_po_2;
    assign n_1_2_po_2 = (n_0_2) << 5;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_2_sum_pos;
    assign n_1_2_sum_pos = 13'b1000000000000;
    wire [15:0] n_1_2_sum_neg;
    assign n_1_2_sum_neg = n_1_2_po_1 + n_1_2_po_2;
    wire signed [16:0] n_1_2_sum;
    assign n_1_2_sum = $signed({1'b0,n_1_2_sum_pos}) - $signed({1'b0,n_1_2_sum_neg});

    wire signed [16:0] n_1_2_sum_f;
    assign n_1_2_sum_f = $signed(n_1_2_sum);
    //relu
    wire [15:0] n_1_2;
    assign n_1_2 = (n_1_2_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_2_sum_f);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [15:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [15:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

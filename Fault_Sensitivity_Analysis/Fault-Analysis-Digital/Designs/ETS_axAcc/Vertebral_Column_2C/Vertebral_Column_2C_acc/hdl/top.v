//weights: [[[32, -32, 8, 64, -64, 64], [-32, 32, -8, -8, -16, 64], [16, 32, 4, -8, -32, 64]], [[16, 64, 8], [-16, -64, 0]]]
//intercepts: [[32, 32, -64], [-64, 64]]
module top (inp, out);
input [23:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight 32 : 6'b100000
    wire [9:0] n_0_0_po_0;
    assign n_0_0_po_0 = (inp[3:0]) << 5;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_0_po_1;
    assign n_0_0_po_1 = (inp[7:4]) << 5;

    //weight 8 : 4'b1000
    wire [7:0] n_0_0_po_2;
    assign n_0_0_po_2 = (inp[11:8]) << 3;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_0_po_3;
    assign n_0_0_po_3 = (inp[15:12]) << 6;

    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_0_po_4;
    assign n_0_0_po_4 = (inp[19:16]) << 6;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_0_po_5;
    assign n_0_0_po_5 = (inp[23:20]) << 6;

    //accumulate positive/negative subproducts
    wire [12:0] n_0_0_sum_pos;
    assign n_0_0_sum_pos = 8'b10000000 + n_0_0_po_0 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_5;
    wire [12:0] n_0_0_sum_neg;
    assign n_0_0_sum_neg = n_0_0_po_1 + n_0_0_po_4;
    wire signed [13:0] n_0_0_sum;
    assign n_0_0_sum = $signed({1'b0,n_0_0_sum_pos}) - $signed({1'b0,n_0_0_sum_neg});

    wire signed [13:0] n_0_0_sum_f;
    assign n_0_0_sum_f = $signed(n_0_0_sum);
    //relu
    wire [7:0] n_0_0, n_0_0_qrelu;
    DW01_satrnd #(13, 9, 2) USR_n_0_0 ( .din(n_0_0_sum_f[12:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_0_qrelu));
    assign n_0_0 = (n_0_0_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_0_qrelu);

// layer: 0 - neuron: 1
    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_1_po_0;
    assign n_0_1_po_0 = (inp[3:0]) << 5;

    //weight 32 : 6'b100000
    wire [9:0] n_0_1_po_1;
    assign n_0_1_po_1 = (inp[7:4]) << 5;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_1_po_2;
    assign n_0_1_po_2 = (inp[11:8]) << 3;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_1_po_3;
    assign n_0_1_po_3 = (inp[15:12]) << 3;

    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_4;
    assign n_0_1_po_4 = (inp[19:16]) << 4;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_5;
    assign n_0_1_po_5 = (inp[23:20]) << 6;

    //accumulate positive/negative subproducts
    wire [11:0] n_0_1_sum_pos;
    assign n_0_1_sum_pos = 8'b10000000 + n_0_1_po_1 + n_0_1_po_5;
    wire [11:0] n_0_1_sum_neg;
    assign n_0_1_sum_neg = n_0_1_po_0 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4;
    wire signed [12:0] n_0_1_sum;
    assign n_0_1_sum = $signed({1'b0,n_0_1_sum_pos}) - $signed({1'b0,n_0_1_sum_neg});

    wire signed [12:0] n_0_1_sum_f;
    assign n_0_1_sum_f = $signed(n_0_1_sum);
    //relu
    wire [7:0] n_0_1, n_0_1_qrelu;
    DW01_satrnd #(12, 9, 2) USR_n_0_1 ( .din(n_0_1_sum_f[11:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_1_qrelu));
    assign n_0_1 = (n_0_1_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_1_qrelu);

// layer: 0 - neuron: 2
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_0;
    assign n_0_2_po_0 = (inp[3:0]) << 4;

    //weight 32 : 6'b100000
    wire [9:0] n_0_2_po_1;
    assign n_0_2_po_1 = (inp[7:4]) << 5;

    //weight 4 : 3'b100
    wire [6:0] n_0_2_po_2;
    assign n_0_2_po_2 = (inp[11:8]) << 2;

    //weight abs(-8) : 4'b1000
    wire [7:0] n_0_2_po_3;
    assign n_0_2_po_3 = (inp[15:12]) << 3;

    //weight abs(-32) : 6'b100000
    wire [9:0] n_0_2_po_4;
    assign n_0_2_po_4 = (inp[19:16]) << 5;

    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_5;
    assign n_0_2_po_5 = (inp[23:20]) << 6;

    //accumulate positive/negative subproducts
    wire [11:0] n_0_2_sum_pos;
    assign n_0_2_sum_pos = n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_5;
    wire [11:0] n_0_2_sum_neg;
    assign n_0_2_sum_neg = 9'b100000000 + n_0_2_po_3 + n_0_2_po_4;
    wire signed [12:0] n_0_2_sum;
    assign n_0_2_sum = $signed({1'b0,n_0_2_sum_pos}) - $signed({1'b0,n_0_2_sum_neg});

    wire signed [12:0] n_0_2_sum_f;
    assign n_0_2_sum_f = $signed(n_0_2_sum);
    //relu
    wire [7:0] n_0_2, n_0_2_qrelu;
    DW01_satrnd #(12, 9, 2) USR_n_0_2 ( .din(n_0_2_sum_f[11:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_2_qrelu));
    assign n_0_2 = (n_0_2_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_2_qrelu);

// layer: 1 - neuron: 0
    //weight 16 : 5'b10000
    wire [12:0] n_1_0_po_0;
    assign n_1_0_po_0 = (n_0_0) << 4;

    //weight 64 : 7'b1000000
    wire [14:0] n_1_0_po_1;
    assign n_1_0_po_1 = (n_0_1) << 6;

    //weight 8 : 4'b1000
    wire [11:0] n_1_0_po_2;
    assign n_1_0_po_2 = (n_0_2) << 3;

    //accumulate positive/negative subproducts
    wire [15:0] n_1_0_sum_pos;
    assign n_1_0_sum_pos = n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    wire [15:0] n_1_0_sum_neg;
    assign n_1_0_sum_neg = 11'b10000000000;
    wire signed [16:0] n_1_0_sum;
    assign n_1_0_sum = $signed({1'b0,n_1_0_sum_pos}) - $signed({1'b0,n_1_0_sum_neg});

    wire signed [16:0] n_1_0_sum_f;
    assign n_1_0_sum_f = $signed(n_1_0_sum);
    //relu
    wire [15:0] n_1_0;
    assign n_1_0 = (n_1_0_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_0_sum_f);

// layer: 1 - neuron: 1
    //weight abs(-16) : 5'b10000
    wire [12:0] n_1_1_po_0;
    assign n_1_1_po_0 = (n_0_0) << 4;

    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_1_po_1;
    assign n_1_1_po_1 = (n_0_1) << 6;

    //weight 0 : skip
    //accumulate positive/negative subproducts
    wire [15:0] n_1_1_sum_pos;
    assign n_1_1_sum_pos = 11'b10000000000;
    wire [15:0] n_1_1_sum_neg;
    assign n_1_1_sum_neg = n_1_1_po_0 + n_1_1_po_1;
    wire signed [16:0] n_1_1_sum;
    assign n_1_1_sum = $signed({1'b0,n_1_1_sum_pos}) - $signed({1'b0,n_1_1_sum_neg});

    wire signed [16:0] n_1_1_sum_f;
    assign n_1_1_sum_f = $signed(n_1_1_sum);
    //relu
    wire [15:0] n_1_1;
    assign n_1_1 = (n_1_1_sum_f<0) ? $unsigned({16{1'b0}}) : $unsigned(n_1_1_sum_f);

// argmax: 2 classes, need 2 bits
// argmax inp: n_1_0, n_1_1
    //comp level 0
    wire cmp_0_0;
    wire [15:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    assign out = argmax_idx_0_0;

endmodule

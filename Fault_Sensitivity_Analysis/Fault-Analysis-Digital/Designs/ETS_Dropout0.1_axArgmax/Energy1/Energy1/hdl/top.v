//weights: [[[0, 0, 0, 0, 0, 0, 0, 0], [64, -64, 64, 64, 64, -16, 16, -16], [-64, 8, -64, -64, 64, 4, -64, 16]], [[0, -64, 16], [0, 4, 32], [0, 16, -32]]]
//intercepts: [[-64, -64, 64], [64, -64, -64]]
module top (inp, out);
input [31:0] inp;
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
    //accumulate positive/negative subproducts

    //WARN: only negative weights with relu. Using zero
    wire signed [10:0] n_0_0_sum;
    assign n_0_0_sum = $signed({11{1'b0}});

    wire signed [10:0] n_0_0_sum_f;
    assign n_0_0_sum_f = $signed(n_0_0_sum);
    //relu
    wire [7:0] n_0_0, n_0_0_qrelu;
    assign n_0_0 = $unsigned({8{1'b0}});

// layer: 0 - neuron: 1
    wire [3:0] n_0_1_po_0_a;
    assign n_0_1_po_0_a = inp[3:0];
    //mask 8 : [3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_0;
    assign n_0_1_po_0 = ((n_0_1_po_0_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_1_a;
    assign n_0_1_po_1_a = inp[7:4];
    //mask 0 : []
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_1_po_1;
    assign n_0_1_po_1 = (4'd0) << 6;

    wire [3:0] n_0_1_po_2_a;
    assign n_0_1_po_2_a = inp[11:8];
    //mask 9 : [0, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_2;
    assign n_0_1_po_2 = (n_0_1_po_2_a[0] + (n_0_1_po_2_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_3_a;
    assign n_0_1_po_3_a = inp[15:12];
    //mask 0 : []
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_3;
    assign n_0_1_po_3 = (4'd0) << 6;

    wire [3:0] n_0_1_po_4_a;
    assign n_0_1_po_4_a = inp[19:16];
    //mask 9 : [0, 3]
    //weight 64 : 7'b1000000
    wire [10:0] n_0_1_po_4;
    assign n_0_1_po_4 = (n_0_1_po_4_a[0] + (n_0_1_po_4_a[3] << 3)) << 6;

    wire [3:0] n_0_1_po_5_a;
    assign n_0_1_po_5_a = inp[23:20];
    //mask 0 : []
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_5;
    assign n_0_1_po_5 = (4'd0) << 4;

    wire [3:0] n_0_1_po_6_a;
    assign n_0_1_po_6_a = inp[27:24];
    //mask 0 : []
    //weight 16 : 5'b10000
    wire [8:0] n_0_1_po_6;
    assign n_0_1_po_6 = (4'd0) << 4;

    wire [3:0] n_0_1_po_7_a;
    assign n_0_1_po_7_a = inp[31:28];
    //mask 0 : []
    //weight abs(-16) : 5'b10000
    wire [8:0] n_0_1_po_7;
    assign n_0_1_po_7 = (4'd0) << 4;

    //accumulate positive/negative subproducts
    wire [13:0] n_0_1_sum_pos;
    assign n_0_1_sum_pos = n_0_1_po_0 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_6;
    wire [13:0] n_0_1_sum_neg;
    assign n_0_1_sum_neg = 10'b1000000000 + n_0_1_po_1 + n_0_1_po_5 + n_0_1_po_7;
    wire signed [14:0] n_0_1_sum;
    assign n_0_1_sum = $signed({1'b0,n_0_1_sum_pos}) - $signed({1'b0,n_0_1_sum_neg});

    wire signed [14:0] n_0_1_sum_f;
    assign n_0_1_sum_f = $signed(n_0_1_sum);
    //relu
    wire [7:0] n_0_1, n_0_1_qrelu;
    DW01_satrnd #(14, 9, 2) USR_n_0_1 ( .din(n_0_1_sum_f[13:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_1_qrelu));
    assign n_0_1 = (n_0_1_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_1_qrelu);

// layer: 0 - neuron: 2
    wire [3:0] n_0_2_po_0_a;
    assign n_0_2_po_0_a = inp[3:0];
    //mask 0 : []
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_0;
    assign n_0_2_po_0 = (4'd0) << 6;

    wire [3:0] n_0_2_po_1_a;
    assign n_0_2_po_1_a = inp[7:4];
    //mask 3 : [0, 1]
    //weight 8 : 4'b1000
    wire [7:0] n_0_2_po_1;
    assign n_0_2_po_1 = (n_0_2_po_1_a[0] + (n_0_2_po_1_a[1] << 1)) << 3;

    wire [3:0] n_0_2_po_2_a;
    assign n_0_2_po_2_a = inp[11:8];
    //mask 8 : [3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_2;
    assign n_0_2_po_2 = ((n_0_2_po_2_a[3] << 3)) << 6;

    wire [3:0] n_0_2_po_3_a;
    assign n_0_2_po_3_a = inp[15:12];
    //mask 10 : [1, 3]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_3;
    assign n_0_2_po_3 = ((n_0_2_po_3_a[1] << 1) + (n_0_2_po_3_a[3] << 3)) << 6;

    wire [3:0] n_0_2_po_4_a;
    assign n_0_2_po_4_a = inp[19:16];
    //mask 0 : []
    //weight 64 : 7'b1000000
    wire [10:0] n_0_2_po_4;
    assign n_0_2_po_4 = (4'd0) << 6;

    wire [3:0] n_0_2_po_5_a;
    assign n_0_2_po_5_a = inp[23:20];
    //mask 0 : []
    //weight 4 : 3'b100
    wire [6:0] n_0_2_po_5;
    assign n_0_2_po_5 = (4'd0) << 2;

    wire [3:0] n_0_2_po_6_a;
    assign n_0_2_po_6_a = inp[27:24];
    //mask 1 : [0]
    //weight abs(-64) : 7'b1000000
    wire [10:0] n_0_2_po_6;
    assign n_0_2_po_6 = (n_0_2_po_6_a[0]) << 6;

    wire [3:0] n_0_2_po_7_a;
    assign n_0_2_po_7_a = inp[31:28];
    //mask 0 : []
    //weight 16 : 5'b10000
    wire [8:0] n_0_2_po_7;
    assign n_0_2_po_7 = (4'd0) << 4;

    //accumulate positive/negative subproducts
    wire [12:0] n_0_2_sum_pos;
    assign n_0_2_sum_pos = 10'b1000000000 + n_0_2_po_1 + n_0_2_po_4 + n_0_2_po_5 + n_0_2_po_7;
    wire [12:0] n_0_2_sum_neg;
    assign n_0_2_sum_neg = n_0_2_po_0 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_6;
    wire signed [13:0] n_0_2_sum;
    assign n_0_2_sum = $signed({1'b0,n_0_2_sum_pos}) - $signed({1'b0,n_0_2_sum_neg});

    wire signed [13:0] n_0_2_sum_f;
    assign n_0_2_sum_f = $signed(n_0_2_sum);
    //relu
    wire [7:0] n_0_2, n_0_2_qrelu;
    DW01_satrnd #(13, 9, 2) USR_n_0_2 ( .din(n_0_2_sum_f[12:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(n_0_2_qrelu));
    assign n_0_2 = (n_0_2_sum_f<0) ? $unsigned({8{1'b0}}) : $unsigned(n_0_2_qrelu);

// layer: 1 - neuron: 0
    //weight 0 : skip
    wire [7:0] n_1_0_po_1_a;
    assign n_1_0_po_1_a = n_0_1;
    //mask 175 : [0, 1, 2, 3, 5, 7]
    //weight abs(-64) : 7'b1000000
    wire [14:0] n_1_0_po_1;
    assign n_1_0_po_1 = (n_1_0_po_1_a[0] + (n_1_0_po_1_a[1] << 1) + (n_1_0_po_1_a[2] << 2) + (n_1_0_po_1_a[3] << 3) + (n_1_0_po_1_a[5] << 5) + (n_1_0_po_1_a[7] << 7)) << 6;

    wire [7:0] n_1_0_po_2_a;
    assign n_1_0_po_2_a = n_0_2;
    //mask 8 : [3]
    //weight 16 : 5'b10000
    wire [12:0] n_1_0_po_2;
    assign n_1_0_po_2 = ((n_1_0_po_2_a[3] << 3)) << 4;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_0_sum_pos;
    assign n_1_0_sum_pos = 12'b100000000000 + n_1_0_po_2;
    wire [14:0] n_1_0_sum_neg;
    assign n_1_0_sum_neg = n_1_0_po_1;
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
    //mask 1 : [0]
    //weight 4 : 3'b100
    wire [10:0] n_1_1_po_1;
    assign n_1_1_po_1 = (n_1_1_po_1_a[0]) << 2;

    wire [7:0] n_1_1_po_2_a;
    assign n_1_1_po_2_a = n_0_2;
    //mask 177 : [0, 4, 5, 7]
    //weight 32 : 6'b100000
    wire [13:0] n_1_1_po_2;
    assign n_1_1_po_2 = (n_1_1_po_2_a[0] + (n_1_1_po_2_a[4] << 4) + (n_1_1_po_2_a[5] << 5) + (n_1_1_po_2_a[7] << 7)) << 5;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_1_sum_pos;
    assign n_1_1_sum_pos = n_1_1_po_1 + n_1_1_po_2;
    wire [14:0] n_1_1_sum_neg;
    assign n_1_1_sum_neg = 12'b100000000000;
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
    //mask 190 : [1, 2, 3, 4, 5, 7]
    //weight 16 : 5'b10000
    wire [12:0] n_1_2_po_1;
    assign n_1_2_po_1 = ((n_1_2_po_1_a[1] << 1) + (n_1_2_po_1_a[2] << 2) + (n_1_2_po_1_a[3] << 3) + (n_1_2_po_1_a[4] << 4) + (n_1_2_po_1_a[5] << 5) + (n_1_2_po_1_a[7] << 7)) << 4;

    wire [7:0] n_1_2_po_2_a;
    assign n_1_2_po_2_a = n_0_2;
    //mask 0 : []
    //weight abs(-32) : 6'b100000
    wire [13:0] n_1_2_po_2;
    assign n_1_2_po_2 = (8'd0) << 5;

    //accumulate positive/negative subproducts
    wire [14:0] n_1_2_sum_pos;
    assign n_1_2_sum_pos = n_1_2_po_1;
    wire [14:0] n_1_2_sum_neg;
    assign n_1_2_sum_neg = 12'b100000000000 + n_1_2_po_2;
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
    //current mask = 000100000000000
    assign {cmp_0_0} = ( {n_1_0[11]} >= {n_1_1[11]} );
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
    //current mask = 000100000000000
    assign {cmp_1_0} = ( {argmax_val_0_2[11]} >= {argmax_val_0_0[11]} );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_2 : argmax_val_0_0;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_2 : argmax_idx_0_0;

    assign out = argmax_idx_1_0;

endmodule

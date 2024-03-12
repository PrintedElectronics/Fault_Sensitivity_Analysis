//weights: [[[-93, -27, -55, -80, 84, -10, 18, 15], [-18, -22, 5, -14, 80, -24, 64, 43], [2, 16, -36, 58, -43, -19, -66, 8]], [[4, -25, 42], [68, -23, -25], [-102, 37, -46]]]
//intercepts: [[473, -912, 407], [-186, 669, -3346]]
module top (inp, out);
input [31:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight -93 : 8'sb10100011
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb10100011;

    wire signed [11:0] n_0_0_po_1;
    //weight -27 : 8'sb11100101
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11100101;

    wire signed [11:0] n_0_0_po_2;
    //weight -55 : 8'sb11001001
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11001001;

    wire signed [11:0] n_0_0_po_3;
    //weight -80 : 8'sb10110000
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb10110000;

    wire signed [11:0] n_0_0_po_4;
    //weight 84 : 8'sb01010100
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb01010100;

    wire signed [11:0] n_0_0_po_5;
    //weight -10 : 8'sb11110110
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11110110;

    wire signed [11:0] n_0_0_po_6;
    //weight 18 : 8'sb00010010
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb00010010;

    wire signed [11:0] n_0_0_po_7;
    //weight 15 : 8'sb00001111
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00001111;

    wire signed [19:0] n_0_0_sum;
    assign n_0_0_sum = 473 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_5 + n_0_0_po_6 + n_0_0_po_7;
    //relu
    wire [14:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight -18 : 8'sb11101110
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11101110;

    wire signed [11:0] n_0_1_po_1;
    //weight -22 : 8'sb11101010
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11101010;

    wire signed [11:0] n_0_1_po_2;
    //weight 5 : 8'sb00000101
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00000101;

    wire signed [11:0] n_0_1_po_3;
    //weight -14 : 8'sb11110010
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11110010;

    wire signed [11:0] n_0_1_po_4;
    //weight 80 : 8'sb01010000
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb01010000;

    wire signed [11:0] n_0_1_po_5;
    //weight -24 : 8'sb11101000
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11101000;

    wire signed [11:0] n_0_1_po_6;
    //weight 64 : 8'sb01000000
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb01000000;

    wire signed [11:0] n_0_1_po_7;
    //weight 43 : 8'sb00101011
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00101011;

    wire signed [19:0] n_0_1_sum;
    assign n_0_1_sum = -912 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_5 + n_0_1_po_6 + n_0_1_po_7;
    //relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight 2 : 8'sb00000010
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00000010;

    wire signed [11:0] n_0_2_po_1;
    //weight 16 : 8'sb00010000
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00010000;

    wire signed [11:0] n_0_2_po_2;
    //weight -36 : 8'sb11011100
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11011100;

    wire signed [11:0] n_0_2_po_3;
    //weight 58 : 8'sb00111010
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00111010;

    wire signed [11:0] n_0_2_po_4;
    //weight -43 : 8'sb11010101
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11010101;

    wire signed [11:0] n_0_2_po_5;
    //weight -19 : 8'sb11101101
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11101101;

    wire signed [11:0] n_0_2_po_6;
    //weight -66 : 8'sb10111110
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb10111110;

    wire signed [11:0] n_0_2_po_7;
    //weight 8 : 8'sb00001000
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00001000;

    wire signed [19:0] n_0_2_sum;
    assign n_0_2_sum = 407 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_5 + n_0_2_po_6 + n_0_2_po_7;
    //relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [21:0] n_1_0_po_0;
    //weight 4 : 8'sb00000100
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000100;

    wire signed [21:0] n_1_0_po_1;
    //weight -25 : 8'sb11100111
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb11100111;

    wire signed [21:0] n_1_0_po_2;
    //weight 42 : 8'sb00101010
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb00101010;

    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = -186 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [21:0] n_1_1_po_0;
    //weight 68 : 8'sb01000100
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb01000100;

    wire signed [21:0] n_1_1_po_1;
    //weight -23 : 8'sb11101001
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb11101001;

    wire signed [21:0] n_1_1_po_2;
    //weight -25 : 8'sb11100111
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb11100111;

    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = 669 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2;
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [21:0] n_1_2_po_0;
    //weight -102 : 8'sb10011010
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb10011010;

    wire signed [21:0] n_1_2_po_1;
    //weight 37 : 8'sb00100101
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb00100101;

    wire signed [21:0] n_1_2_po_2;
    //weight -46 : 8'sb11010010
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb11010010;

    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = -3346 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [22:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [23:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [23:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

//weights: [[[-13, 22, -8, -17, -30, 38], [-2, -2, 2, 1, -2, -3], [-3, -1, -1, -3, -15, 88]], [[7, 0, -31], [-7, 3, -50], [-2, 1, 51]]]
//intercepts: [[331, -51, -71], [-903, 750, -759]]
module top (inp, out);
input [23:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight -13 : 8'sb11110011
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11110011;

    wire signed [11:0] n_0_0_po_1;
    //weight 22 : 8'sb00010110
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00010110;

    wire signed [11:0] n_0_0_po_2;
    //weight -8 : 8'sb11111000
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11111000;

    wire signed [11:0] n_0_0_po_3;
    //weight -17 : 8'sb11101111
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11101111;

    wire signed [11:0] n_0_0_po_4;
    //weight -30 : 8'sb11100010
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11100010;

    wire signed [11:0] n_0_0_po_5;
    //weight 38 : 8'sb00100110
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb00100110;

    wire signed [17:0] n_0_0_sum;
    assign n_0_0_sum = 331 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_5;
    //relu
    wire [13:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({14{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight -2 : 8'sb11111110
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11111110;

    wire signed [11:0] n_0_1_po_1;
    //weight -2 : 8'sb11111110
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111110;

    wire signed [11:0] n_0_1_po_2;
    //weight 2 : 8'sb00000010
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00000010;

    wire signed [11:0] n_0_1_po_3;
    //weight 1 : 8'sb00000001
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00000001;

    wire signed [11:0] n_0_1_po_4;
    //weight -2 : 8'sb11111110
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11111110;

    wire signed [11:0] n_0_1_po_5;
    //weight -3 : 8'sb11111101
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11111101;

    wire signed [17:0] n_0_1_sum;
    assign n_0_1_sum = -51 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_5;
    //relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight -3 : 8'sb11111101
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11111101;

    wire signed [11:0] n_0_2_po_1;
    //weight -1 : 8'sb11111111
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111111;

    wire signed [11:0] n_0_2_po_2;
    //weight -1 : 8'sb11111111
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11111111;

    wire signed [11:0] n_0_2_po_3;
    //weight -3 : 8'sb11111101
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11111101;

    wire signed [11:0] n_0_2_po_4;
    //weight -15 : 8'sb11110001
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11110001;

    wire signed [11:0] n_0_2_po_5;
    //weight 88 : 8'sb01011000
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb01011000;

    wire signed [17:0] n_0_2_sum;
    assign n_0_2_sum = -71 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_5;
    //relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [19:0] n_1_0_po_0;
    //weight 7 : 8'sb00000111
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000111;

    //weight 0 : skip
    wire signed [19:0] n_1_0_po_2;
    //weight -31 : 8'sb11100001
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb11100001;

    wire signed [22:0] n_1_0_sum;
    assign n_1_0_sum = -903 + n_1_0_po_0 + n_1_0_po_2;
    //relu
    wire [20:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({21{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [19:0] n_1_1_po_0;
    //weight -7 : 8'sb11111001
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111001;

    wire signed [19:0] n_1_1_po_1;
    //weight 3 : 8'sb00000011
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb00000011;

    wire signed [19:0] n_1_1_po_2;
    //weight -50 : 8'sb11001110
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb11001110;

    wire signed [22:0] n_1_1_sum;
    assign n_1_1_sum = 750 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2;
    //relu
    wire [20:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({21{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [19:0] n_1_2_po_0;
    //weight -2 : 8'sb11111110
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111110;

    wire signed [19:0] n_1_2_po_1;
    //weight 1 : 8'sb00000001
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb00000001;

    wire signed [19:0] n_1_2_po_2;
    //weight 51 : 8'sb00110011
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb00110011;

    wire signed [22:0] n_1_2_sum;
    assign n_1_2_sum = -759 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [20:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({21{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [20:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [20:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

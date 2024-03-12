//weights: [[[88, 86, -88, -86], [59, 57, -59, -59], [-12, -3, -6, -12]], [[-98, 72, 12], [1, 55, -4], [33, -72, 11]]]
//intercepts: [[-1, 571, -164], [-38551, -33633, 33375]]
module top (inp, out);
input [15:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight 88 : 8'sb01011000
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb01011000;

    wire signed [11:0] n_0_0_po_1;
    //weight 86 : 8'sb01010110
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb01010110;

    wire signed [11:0] n_0_0_po_2;
    //weight -88 : 8'sb10101000
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb10101000;

    wire signed [11:0] n_0_0_po_3;
    //weight -86 : 8'sb10101010
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb10101010;

    wire signed [15:0] n_0_0_sum;
    assign n_0_0_sum = -1 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3;
    //relu
    wire [12:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({13{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight 59 : 8'sb00111011
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00111011;

    wire signed [11:0] n_0_1_po_1;
    //weight 57 : 8'sb00111001
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00111001;

    wire signed [11:0] n_0_1_po_2;
    //weight -59 : 8'sb11000101
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11000101;

    wire signed [11:0] n_0_1_po_3;
    //weight -59 : 8'sb11000101
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11000101;

    wire signed [15:0] n_0_1_sum;
    assign n_0_1_sum = 571 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3;
    //relu
    wire [13:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({14{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight -12 : 8'sb11110100
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11110100;

    wire signed [11:0] n_0_2_po_1;
    //weight -3 : 8'sb11111101
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111101;

    wire signed [11:0] n_0_2_po_2;
    //weight -6 : 8'sb11111010
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11111010;

    wire signed [11:0] n_0_2_po_3;
    //weight -12 : 8'sb11110100
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11110100;

    wire signed [15:0] n_0_2_sum;
    assign n_0_2_sum = -164 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3;
    //relu
    wire [21:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [20:0] n_1_0_po_0;
    //weight -98 : 8'sb10011110
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb10011110;

    wire signed [20:0] n_1_0_po_1;
    //weight 72 : 8'sb01001000
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb01001000;

    wire signed [20:0] n_1_0_po_2;
    //weight 12 : 8'sb00001100
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb00001100;

    wire signed [23:0] n_1_0_sum;
    assign n_1_0_sum = -38551 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [21:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [20:0] n_1_1_po_0;
    //weight 1 : 8'sb00000001
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000001;

    wire signed [20:0] n_1_1_po_1;
    //weight 55 : 8'sb00110111
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb00110111;

    wire signed [20:0] n_1_1_po_2;
    //weight -4 : 8'sb11111100
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb11111100;

    wire signed [23:0] n_1_1_sum;
    assign n_1_1_sum = -33633 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2;
    //relu
    wire [21:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [20:0] n_1_2_po_0;
    //weight 33 : 8'sb00100001
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb00100001;

    wire signed [20:0] n_1_2_po_1;
    //weight -72 : 8'sb10111000
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb10111000;

    wire signed [20:0] n_1_2_po_2;
    //weight 11 : 8'sb00001011
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb00001011;

    wire signed [23:0] n_1_2_sum;
    assign n_1_2_sum = 33375 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [21:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [22:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [22:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

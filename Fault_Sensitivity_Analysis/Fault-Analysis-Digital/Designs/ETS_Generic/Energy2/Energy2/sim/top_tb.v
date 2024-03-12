`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=8;
    parameter WIDTH_A=4;
    parameter NUM_W=33;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=11;
    parameter NUM_B1=3;
    parameter WIDTH_B1=13;

localparam period = 320000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [-46, 37, -102, -25, -23, 68, 42, -25, 4, 8, -66, -19, -43, 58, -36, 16, 2, 43, 64, -24, 80, -14, 5, -22, -18, 15, 18, -10, 84, -80, -55, -27, -93]
assign weights = {8'sb11010010, 8'sb00100101, 8'sb10011010, 8'sb11100111, 8'sb11101001, 8'sb01000100, 8'sb00101010, 8'sb11100111, 8'sb00000100, 8'sb00001000, 8'sb10111110, 8'sb11101101, 8'sb11010101, 8'sb00111010, 8'sb11011100, 8'sb00010000, 8'sb00000010, 8'sb00101011, 8'sb01000000, 8'sb11101000, 8'sb01010000, 8'sb11110010, 8'sb00000101, 8'sb11101010, 8'sb11101110, 8'sb00001111, 8'sb00010010, 8'sb11110110, 8'sb01010100, 8'sb10110000, 8'sb11001001, 8'sb11100101, 8'sb10100011};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [-3346, 669, -186, 407, -912, 473]
assign biases = {13'sb1001011101110, 13'sb0001010011101, 13'sb1111101000110, 11'sb00110010111, 11'sb10001110000, 11'sb00111011001};
top DUT(.inp(inp),
        .weights(weights),
        .biases(biases),
        .out(out)
        );


    integer inFile,outFile,i;
    initial
    begin
        $display($time, " << Starting the Simulation >>");
            inFile = $fopen("./sim/sim.Xtest","r");
        if (inFile == `NULL) begin
                $display($time, " file not found");
                $finish;
        end
        outFile = $fopen("./sim/output.txt");
        while (!$feof(inFile)) begin
            for (i=0;i<NUM_A;i=i+1) begin
                $fscanf(inFile,"%d ",at[i]);
            end
            $fscanf(inFile,"\n");
            #(period)
            $fwrite(outFile,"%d\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end


    genvar gi;
    generate
    for (gi=0;gi<NUM_A;gi=gi+1) begin : genbit
        assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    end
    endgenerate


    endmodule

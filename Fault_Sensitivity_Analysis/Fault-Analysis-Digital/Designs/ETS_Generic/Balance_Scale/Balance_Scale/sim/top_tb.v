`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=4;
    parameter WIDTH_A=4;
    parameter NUM_W=21;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=11;
    parameter NUM_B1=3;
    parameter WIDTH_B1=17;

localparam period = 320000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [11, -72, 33, -4, 55, 1, 12, 72, -98, -12, -6, -3, -12, -59, -59, 57, 59, -86, -88, 86, 88]
assign weights = {8'sb00001011, 8'sb10111000, 8'sb00100001, 8'sb11111100, 8'sb00110111, 8'sb00000001, 8'sb00001100, 8'sb01001000, 8'sb10011110, 8'sb11110100, 8'sb11111010, 8'sb11111101, 8'sb11110100, 8'sb11000101, 8'sb11000101, 8'sb00111001, 8'sb00111011, 8'sb10101010, 8'sb10101000, 8'sb01010110, 8'sb01011000};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [33375, -33633, -38551, -164, 571, -1]
assign biases = {17'sb01000001001011111, 17'sb10111110010011111, 17'sb10110100101101001, 11'sb11101011100, 11'sb01000111011, 11'sb11111111111};
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

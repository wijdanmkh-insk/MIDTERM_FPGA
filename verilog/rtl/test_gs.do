transcript on

# Clean previous work library
if {[file exists work]} {
    vdel -lib work -all
}

# Create work library
vlib work

# Compile grayscale CNN and testbench
vlog -sv cnn_top_gs.v
vlog -sv tb_cnn_top_gs.v

# Load testbench
vsim -voptargs=+acc work.tb_cnn_top

# Run simulation
run -all

# Exit Questa
quit -f

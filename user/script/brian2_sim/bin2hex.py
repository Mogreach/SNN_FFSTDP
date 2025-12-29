with open("all_spikes.bin", "rb") as bin_file, open("all_spikes.hex", "w") as hex_file:
    byte = bin_file.read(1)
    while byte:
        hex_file.write(f"{int.from_bytes(byte, 'big'):02X}\n")
        byte = bin_file.read(1)

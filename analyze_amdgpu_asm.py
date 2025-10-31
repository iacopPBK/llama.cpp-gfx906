#!/usr/bin/env python3
"""
AMD GPU Assembly Analyzer
Analyzes AMDGPU assembly (.s files) to find optimization opportunities and bottlenecks.

Features:
- Detects unfused DPP instruction opportunities
- Analyzes barrier/synchronization overhead
- Estimates register pressure
- Identifies memory access patterns
- Compares instruction counts between versions

Usage:
    python analyze_amdgpu_asm.py <assembly_file.s>
    python analyze_amdgpu_asm.py <file1.s> <file2.s>  # Compare two versions
"""

import sys
import re
from collections import defaultdict, Counter
from pathlib import Path

class AMDGPUAssemblyAnalyzer:
    def __init__(self, asm_file):
        self.file_path = Path(asm_file)
        self.lines = self.file_path.read_text().splitlines()
        self.kernels = {}
        self.current_kernel = None

    def parse_kernels(self):
        """Parse assembly into individual kernels."""
        kernel_pattern = re.compile(r'^(\S+):.*@\s*(\S+)')
        current_kernel_name = None
        current_kernel_lines = []

        for line in self.lines:
            match = kernel_pattern.match(line.strip())
            if match:
                # Save previous kernel
                if current_kernel_name:
                    self.kernels[current_kernel_name] = current_kernel_lines
                current_kernel_name = match.group(2) or match.group(1)
                current_kernel_lines = [line]
            elif current_kernel_name:
                current_kernel_lines.append(line)

        # Save last kernel
        if current_kernel_name:
            self.kernels[current_kernel_name] = current_kernel_lines

    def find_unfused_dpp_patterns(self):
        """Find v_mov_b32_dpp followed by ALU operations that could be fused."""
        unfused_patterns = []

        for kernel_name, lines in self.kernels.items():
            for i, line in enumerate(lines):
                # Check for v_mov_b32_dpp
                if 'v_mov_b32_dpp' in line:
                    dpp_match = re.search(r'v_mov_b32_dpp\s+(\w+),\s*(\w+)\s+(.*)', line)
                    if not dpp_match:
                        continue

                    dest_reg = dpp_match.group(1)
                    src_reg = dpp_match.group(2)
                    dpp_ctrl = dpp_match.group(3)

                    # Look ahead for ALU operation using the destination
                    for j in range(i+1, min(i+5, len(lines))):
                        next_line = lines[j].strip()

                        # Skip empty lines and comments
                        if not next_line or next_line.startswith(';') or next_line.startswith('//'):
                            continue

                        # Check for fuseable operations
                        fuseable_ops = [
                            (r'v_add_f32_e32\s+(\w+),\s*(\w+),\s*(\w+)', 'v_add_f32_dpp'),
                            (r'v_max_f32_e32\s+(\w+),\s*(\w+),\s*(\w+)', 'v_max_f32_dpp'),
                            (r'v_min_f32_e32\s+(\w+),\s*(\w+),\s*(\w+)', 'v_min_f32_dpp'),
                            (r'v_mul_f32_e32\s+(\w+),\s*(\w+),\s*(\w+)', 'v_mul_f32_dpp'),
                        ]

                        for pattern, fused_instr in fuseable_ops:
                            alu_match = re.search(pattern, next_line)
                            if alu_match and dest_reg in [alu_match.group(2), alu_match.group(3)]:
                                unfused_patterns.append({
                                    'kernel': kernel_name,
                                    'line': i,
                                    'dpp_line': line.strip(),
                                    'alu_line': next_line,
                                    'fused_instruction': fused_instr,
                                    'dpp_ctrl': dpp_ctrl,
                                })
                                break
                        break

        return unfused_patterns

    def analyze_barriers(self):
        """Analyze s_nop and s_waitcnt barrier usage."""
        barriers = defaultdict(list)

        for kernel_name, lines in self.kernels.items():
            for i, line in enumerate(lines):
                # s_nop barriers
                nop_match = re.search(r's_nop\s+(\d+)', line)
                if nop_match:
                    cycles = int(nop_match.group(1)) + 1  # s_nop N = N+1 cycles
                    barriers[kernel_name].append({
                        'type': 's_nop',
                        'cycles': cycles,
                        'line': i,
                        'text': line.strip()
                    })

                # s_waitcnt barriers
                if 's_waitcnt' in line:
                    barriers[kernel_name].append({
                        'type': 's_waitcnt',
                        'line': i,
                        'text': line.strip()
                    })

        return barriers

    def count_instruction_types(self):
        """Count different instruction types."""
        counts = defaultdict(lambda: defaultdict(int))

        for kernel_name, lines in self.kernels.items():
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';') or line.startswith('//'):
                    continue

                # Extract instruction mnemonic
                parts = line.split()
                if not parts:
                    continue

                instr = parts[0]

                # Categorize instructions
                if instr.startswith('v_'):
                    if '_dpp' in instr:
                        counts[kernel_name]['fused_dpp'] += 1
                        counts[kernel_name][instr] += 1
                    elif 'v_mov_b32_dpp' in instr:
                        counts[kernel_name]['unfused_dpp'] += 1
                    else:
                        counts[kernel_name]['valu'] += 1
                elif instr.startswith('s_'):
                    if 's_nop' in instr or 's_waitcnt' in instr:
                        counts[kernel_name]['barriers'] += 1
                    else:
                        counts[kernel_name]['salu'] += 1
                elif 'global_load' in instr or 'global_store' in instr:
                    counts[kernel_name]['global_mem'] += 1
                elif 'ds_' in instr:
                    counts[kernel_name]['lds'] += 1

        return counts

    def estimate_register_pressure(self):
        """Estimate VGPR and SGPR usage."""
        reg_usage = defaultdict(lambda: {'vgpr': set(), 'sgpr': set()})

        for kernel_name, lines in self.kernels.items():
            for line in lines:
                # Find VGPR references (v0-v255)
                vgprs = re.findall(r'\bv(\d+)\b', line)
                for vgpr in vgprs:
                    reg_usage[kernel_name]['vgpr'].add(int(vgpr))

                # Find SGPR references (s0-s103)
                sgprs = re.findall(r'\bs(\d+)\b', line)
                for sgpr in sgprs:
                    reg_usage[kernel_name]['sgpr'].add(int(sgpr))

        # Convert sets to max register numbers
        result = {}
        for kernel_name, regs in reg_usage.items():
            result[kernel_name] = {
                'vgpr_count': max(regs['vgpr']) + 1 if regs['vgpr'] else 0,
                'sgpr_count': max(regs['sgpr']) + 1 if regs['sgpr'] else 0,
            }

        return result

    def print_report(self):
        """Generate comprehensive analysis report."""
        print(f"\n{'='*80}")
        print(f"AMD GPU Assembly Analysis: {self.file_path.name}")
        print(f"{'='*80}\n")

        # Parse kernels
        self.parse_kernels()
        print(f"Found {len(self.kernels)} kernels\n")

        # Unfused DPP opportunities
        print(f"\n{'='*80}")
        print("OPTIMIZATION OPPORTUNITY: Unfused DPP Instructions")
        print(f"{'='*80}")
        unfused = self.find_unfused_dpp_patterns()
        if unfused:
            print(f"Found {len(unfused)} unfused DPP patterns that could be optimized:\n")
            for pattern in unfused[:10]:  # Show first 10
                print(f"Kernel: {pattern['kernel']}")
                print(f"  Line {pattern['line']}: {pattern['dpp_line']}")
                print(f"  Followed by: {pattern['alu_line']}")
                print(f"  → Could fuse to: {pattern['fused_instruction']} ... {pattern['dpp_ctrl']}")
                print()
            if len(unfused) > 10:
                print(f"... and {len(unfused) - 10} more\n")
        else:
            print("✓ No unfused DPP opportunities found (all optimized!)\n")

        # Instruction counts
        print(f"\n{'='*80}")
        print("Instruction Type Statistics")
        print(f"{'='*80}")
        counts = self.count_instruction_types()
        for kernel_name, kernel_counts in sorted(counts.items()):
            print(f"\n{kernel_name}:")
            for instr_type, count in sorted(kernel_counts.items()):
                print(f"  {instr_type:20s}: {count:5d}")

        # Barriers
        print(f"\n{'='*80}")
        print("Barrier/Synchronization Analysis")
        print(f"{'='*80}")
        barriers = self.analyze_barriers()
        for kernel_name, barrier_list in sorted(barriers.items()):
            print(f"\n{kernel_name}: {len(barrier_list)} barriers")

            # Calculate total barrier cycles
            total_cycles = sum(b['cycles'] for b in barrier_list if 'cycles' in b)
            nop_count = sum(1 for b in barrier_list if b['type'] == 's_nop')
            waitcnt_count = sum(1 for b in barrier_list if b['type'] == 's_waitcnt')

            print(f"  s_nop barriers: {nop_count} (total ~{total_cycles} cycles)")
            print(f"  s_waitcnt barriers: {waitcnt_count}")

        # Register pressure
        print(f"\n{'='*80}")
        print("Estimated Register Pressure")
        print(f"{'='*80}")
        reg_pressure = self.estimate_register_pressure()
        for kernel_name, regs in sorted(reg_pressure.items()):
            vgpr_pct = (regs['vgpr_count'] / 256) * 100
            sgpr_pct = (regs['sgpr_count'] / 104) * 100
            print(f"\n{kernel_name}:")
            print(f"  VGPRs: {regs['vgpr_count']:3d}/256 ({vgpr_pct:5.1f}%)")
            print(f"  SGPRs: {regs['sgpr_count']:3d}/104 ({sgpr_pct:5.1f}%)")
            if vgpr_pct > 75:
                print(f"  ⚠ WARNING: High VGPR pressure may limit occupancy!")

        print(f"\n{'='*80}\n")

def compare_assemblies(file1, file2):
    """Compare two assembly files to show optimization impact."""
    analyzer1 = AMDGPUAssemblyAnalyzer(file1)
    analyzer2 = AMDGPUAssemblyAnalyzer(file2)

    analyzer1.parse_kernels()
    analyzer2.parse_kernels()

    counts1 = analyzer1.count_instruction_types()
    counts2 = analyzer2.count_instruction_types()

    print(f"\n{'='*80}")
    print(f"COMPARISON: {Path(file1).name} vs {Path(file2).name}")
    print(f"{'='*80}\n")

    # Find common kernels
    common_kernels = set(counts1.keys()) & set(counts2.keys())

    for kernel in sorted(common_kernels):
        c1 = counts1[kernel]
        c2 = counts2[kernel]

        print(f"\n{kernel}:")
        print(f"  {'Metric':<20s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
        print(f"  {'-'*55}")

        all_metrics = set(c1.keys()) | set(c2.keys())
        for metric in sorted(all_metrics):
            v1 = c1.get(metric, 0)
            v2 = c2.get(metric, 0)
            delta = v2 - v1
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            if delta != 0:
                print(f"  {metric:<20s} {v1:>10d} {v2:>10d} {delta_str:>10s}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if len(sys.argv) == 2:
        # Analyze single file
        analyzer = AMDGPUAssemblyAnalyzer(sys.argv[1])
        analyzer.print_report()
    elif len(sys.argv) == 3:
        # Compare two files
        compare_assemblies(sys.argv[1], sys.argv[2])
    else:
        print("Usage: analyze_amdgpu_asm.py <file.s> [file2.s]")
        sys.exit(1)

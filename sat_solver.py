#!/usr/bin/env python3
import sys
import random
import argparse
from collections import defaultdict
import time


# --- DIMACS parser ---
def parse_dimacs(filename):
    clauses = []
    num_vars = 0
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == 'c':
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 3:
                        num_vars = int(parts[2])
                else:
                    lits = list(map(int, line.split()))
                    if lits and lits[-1] == 0:
                        lits.pop()
                    if lits:  # Only add non-empty clauses
                        clauses.append(lits)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        return 0, []  # Fallback on file error
    return num_vars, clauses


# --- SAT Solvers ---

# Resolution Solver
class ResolutionSolver:
    def __init__(self, clauses):
        self.clauses = [frozenset(c) for c in clauses]

    def solve(self):
        known = set(self.clauses)
        new = set()
        while True:
            pairs = [(c1, c2) for i, c1 in enumerate(known)
                     for j, c2 in enumerate(known) if i < j]
            for c1, c2 in pairs:
                for lit in c1:
                    if -lit in c2:
                        resolvent = (c1 | c2) - {lit, -lit}
                        if not resolvent:
                            return False
                        f_res = frozenset(resolvent)
                        if f_res not in known:
                            new.add(f_res)
            if not new:
                return True
            known |= new
            new.clear()


# DP Solver
class DPSolver:
    def __init__(self, clauses):
        self.clauses = [set(c) for c in clauses]

    def solve(self, clauses=None):
        if clauses is None:
            clauses = self.clauses
        if not clauses:
            return True
        if any(len(c) == 0 for c in clauses):
            return False

        # Find a variable that appears in at least one clause
        var = None
        for c in clauses:
            if c:
                var = abs(next(iter(c)))
                break

        if var is None:
            return True  # No variables left, formula is satisfied

        pos = [c for c in clauses if var in c]
        neg = [c for c in clauses if -var in c]
        rest = [c for c in clauses if var not in c and -var not in c]

        resolvents = [set(c1 | c2 - {var, -var}) for c1 in pos for c2 in neg]
        return self.solve(rest + resolvents)


# DPLL Solver
class DPLLSolver:
    def __init__(self, num_vars, clauses, heuristic='random'):
        self.num_vars = num_vars
        self.original_clauses = [set(c) for c in clauses]
        self.heuristic = heuristic

    def unit_propagate(self, clauses, assignment):
        changed = True
        while changed:
            changed = False
            unit_clauses = [c for c in clauses if len(c) == 1]
            if not unit_clauses:
                break

            for c in unit_clauses:
                lit = next(iter(c))
                var, val = abs(lit), lit > 0

                if var in assignment:
                    if assignment[var] != val:
                        return None, None  # Contradiction found
                    continue  # Already assigned correctly

                assignment[var] = val
                changed = True

                # Update clauses
                new_clauses = []
                for clause in clauses:
                    if lit in clause:  # Clause is satisfied
                        continue
                    if -lit in clause:  # Remove literal from clause
                        new_clause = clause - {-lit}
                        if not new_clause:  # Empty clause - contradiction
                            return None, None
                        new_clauses.append(new_clause)
                    else:
                        new_clauses.append(clause)

                clauses = new_clauses

        return clauses, assignment

    def pure_literal_elimination(self, clauses, assignment):
        # Count literal occurrences
        literals = defaultdict(int)
        for c in clauses:
            for lit in c:
                literals[lit] += 1

        changed = False
        new_clauses = list(clauses)  # Make a copy

        for var in range(1, self.num_vars + 1):
            if var not in assignment:
                pos_count = literals[var]
                neg_count = literals[-var]

                if pos_count > 0 and neg_count == 0:  # Pure positive
                    assignment[var] = True
                    new_clauses = [c for c in new_clauses if var not in c]
                    changed = True
                elif neg_count > 0 and pos_count == 0:  # Pure negative
                    assignment[var] = False
                    new_clauses = [c for c in new_clauses if -var not in c]
                    changed = True

        return new_clauses, assignment, changed

    def choose_variable(self, clauses, assignment):
        unassigned = [v for v in range(1, self.num_vars + 1) if v not in assignment]
        if not unassigned:
            return None

        if self.heuristic == 'random':
            return random.choice(unassigned)
        elif self.heuristic == 'jw':
            # Jeroslow-Wang heuristic
            scores = defaultdict(float)
            for v in unassigned:
                for c in clauses:
                    if v in c:
                        scores[v] += 2 ** -len(c)
                    if -v in c:
                        scores[-v] += 2 ** -len(c)
            return max(unassigned, key=lambda v: scores[v] + scores[-v])
        elif self.heuristic == 'mom':
            # Maximum Occurrences in Minimum size clauses
            if not clauses:
                return unassigned[0]

            min_len = min((len(c) for c in clauses), default=0)
            if min_len == 0:
                return unassigned[0]

            counts = defaultdict(int)
            for c in clauses:
                if len(c) == min_len:
                    for lit in c:
                        counts[abs(lit)] += 1

            # Filter to only count unassigned variables
            return max(unassigned, key=lambda v: counts[v] if v in counts else 0)

        return unassigned[0]  # Default

    def solve_recursive(self, clauses, assignment):
        # Unit propagation
        clauses, assignment = self.unit_propagate(clauses, assignment)
        if clauses is None:  # Contradiction found
            return False, {}

        # Pure literal elimination
        clauses, assignment, changed = self.pure_literal_elimination(clauses, assignment)
        if changed:
            return self.solve_recursive(clauses, assignment)

        # Check if solved
        if not clauses:
            return True, assignment
        if any(len(c) == 0 for c in clauses):
            return False, {}

        # Choose variable
        var = self.choose_variable(clauses, assignment)
        if var is None:  # All variables assigned
            return True, assignment

        # Try both values
        for val in [True, False]:
            new_assign = assignment.copy()
            new_assign[var] = val
            lit = var if val else -var

            # Simplify formula
            new_clauses = []
            for c in clauses:
                if lit in c:
                    continue  # Clause is satisfied
                if -lit in c:
                    new_c = c - {-lit}
                    if not new_c:  # Empty clause
                        continue  # Skip this assignment
                    new_clauses.append(new_c)
                else:
                    new_clauses.append(c)

            # Recursive call
            sat, result = self.solve_recursive(new_clauses, new_assign)
            if sat:
                return True, result

        return False, {}

    def solve(self):
        return self.solve_recursive(self.original_clauses, {})


def main():
    parser = argparse.ArgumentParser(description='SAT Solver')
    parser.add_argument('file', help='DIMACS CNF file')
    parser.add_argument('-a', '--algorithm', choices=['resolution', 'dp', 'dpll'], default='dpll',
                        help='Algorithm to use (default: dpll)')
    parser.add_argument('-H', '--heuristic', choices=['random', 'jw', 'mom'], default='mom',
                        help='Variable selection heuristic for DPLL (default: mom)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Parse input file
    if args.verbose:
        print(f"Parsing {args.file}...")
    num_vars, clauses = parse_dimacs(args.file)

    if num_vars == 0 or not clauses:
        print("ERROR: Invalid CNF file or parsing error", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {num_vars} variables and {len(clauses)} clauses")

    # Solve using selected algorithm
    start_time = time.time()

    try:
        if args.algorithm == 'resolution':
            solver = ResolutionSolver(clauses)
            is_sat = solver.solve()
            result = is_sat
        elif args.algorithm == 'dp':
            solver = DPSolver(clauses)
            is_sat = solver.solve()
            result = is_sat
        elif args.algorithm == 'dpll':
            solver = DPLLSolver(num_vars, clauses, args.heuristic)
            is_sat, assignment = solver.solve()
            result = is_sat
        else:
            print(f"ERROR: Unknown algorithm {args.algorithm}", file=sys.stderr)
            return 1

        end_time = time.time()
        elapsed = end_time - start_time

        # Print result
        if result:
            print("SAT")
        else:
            print("UNSAT")

        if args.verbose:
            print(f"Time: {elapsed:.3f} seconds")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
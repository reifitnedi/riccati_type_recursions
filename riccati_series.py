# python riccati_series.py paper
# python riccati_series.py seed
# python riccati_series.py s --r 8
# python riccati_series.py s --max-r 15
# python riccati_series.py t --sign + --q 1 --r 10
# python riccati_series.py t --sign - --q 2 --max-r 8

import argparse
import sys
import sympy as sp

p = sp.Symbol("p")
L = sp.Symbol("L", nonzero=True)
y = sp.Symbol("y", nonzero=True)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def canonical(expr):
    return sp.cancel(sp.together(expr))


def expanded_rational(expr):
    num, den = sp.fraction(canonical(expr))
    return sp.cancel(sp.expand(num) / sp.factor(den))


def build_alpha(max_k):
    a = {0: sp.Integer(0)}
    if max_k >= 1:
        a[1] = -sp.Rational(1, 2) / L
    for k in range(1, max_k):
        first = sum(a[j] * a[k - j] for j in range(1, k))
        second = sum(a[j] * a[k - 1 - j] for j in range(1, k - 1))
        a[k + 1] = canonical(
            -(2 * (k - p) * a[k] + L * first + second) /
            (2 * L * (k + 1))
        )
    return a


def s_formula(r, coeffs=None):
    if r < 2:
        raise ValueError("S_r starts at r=2")
    if coeffs is None or r - 1 not in coeffs:
        coeffs = build_alpha(r - 1)
    return expanded_rational(
        -sp.Rational(1, 2) *
        (L * coeffs[r - 2] + L ** 2 * coeffs[r - 1])
    )


def seed_formula():
    return p + 1


def gamma0(sign, q):
    if sign == "+":
        w = sp.Integer(0)
        for i in range(q - 1):
            nu = p + q - 1 - i
            w = canonical(y / (2 * nu - w))
        return canonical(w / 2)
    a = p - q
    w = 2 * (a + 1)
    for i in range(2, q + 1):
        nu = a + i
        w = canonical(2 * nu - y / w)
    return canonical(w / 2)


def build_gamma(max_k, sign, q):
    g = {0: gamma0(sign, q)}
    for k in range(max_k):
        first = sum(g[j] * g[k - j] for j in range(k + 1))
        second = sum(g[j] * g[k - 1 - j] for j in range(k))
        delta = y / 4 if k == 0 else 0
        g[k + 1] = canonical(
            (first + second - (k + p + 1) * g[k] + delta) /
            (k + 1)
        )
    return g


def t_formula(r, sign, q=1, coeffs=None):
    if r < 1:
        raise ValueError("r must be at least 1")
    if q < 1:
        raise ValueError("q must be at least 1")
    if sign not in {"+", "-"}:
        raise ValueError("sign must be '+' or '-'")
    if coeffs is None or r - 1 not in coeffs:
        coeffs = build_gamma(r - 1, sign, q)
    return expanded_rational(coeffs[r - 1] / y ** r)


def generate_s_range(max_r):
    if max_r < 2:
        raise ValueError("max-r for S must be at least 2")
    coeffs = build_alpha(max_r - 1)
    return [(r, s_formula(r, coeffs)) for r in range(2, max_r + 1)]


def generate_t_range(max_r, sign, q):
    coeffs = build_gamma(max_r - 1, sign, q)
    return [(r, t_formula(r, sign, q, coeffs)) for r in range(1, max_r + 1)]


def print_s(formulas):
    print("L = j_(p,n)^2")
    for r, expr in formulas:
        print(f"S_{r}(p,n) = {sp.sstr(expr)}")


def print_t(formulas, sign, q):
    center = f"p+{q}" if sign == "+" else f"p-{q}"
    print(f"y = j_({center},n)^2")
    for r, expr in formulas:
        print(f"T_{r}^({sign}{q})(p,n) = {sp.sstr(expr)}")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("paper")
    sub.add_parser("seed")
    s_parser = sub.add_parser("s")
    s_group = s_parser.add_mutually_exclusive_group(required=True)
    s_group.add_argument("--r", type=positive_int)
    s_group.add_argument("--max-r", type=positive_int)
    t_parser = sub.add_parser("t")
    t_parser.add_argument("--sign", choices=["+", "-"], required=True)
    t_parser.add_argument("--q", type=positive_int, default=1)
    t_group = t_parser.add_mutually_exclusive_group(required=True)
    t_group.add_argument("--r", type=positive_int)
    t_group.add_argument("--max-r", type=positive_int)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "seed":
            print(f"seed = {sp.sstr(seed_formula())}")
            return 0
        if args.command == "s":
            max_r = args.r if args.r is not None else args.max_r
            formulas = generate_s_range(max_r)
            if args.r is not None:
                formulas = [formulas[-1]]
            print_s(formulas)
            return 0
        if args.command == "t":
            max_r = args.r if args.r is not None else args.max_r
            formulas = generate_t_range(max_r, args.sign, args.q)
            if args.r is not None:
                formulas = [formulas[-1]]
            print_t(formulas, args.sign, args.q)
            return 0
        if args.command == "paper":
            print("Appendix A")
            print_s(generate_s_range(15))
            print()
            print("Appendix B")
            print_t(generate_t_range(10, "+", 1), "+", 1)
            print()
            print("Appendix C")
            print_t(generate_t_range(10, "-", 1), "-", 1)
            return 0
        raise RuntimeError("unknown command")
    except (ValueError, ZeroDivisionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

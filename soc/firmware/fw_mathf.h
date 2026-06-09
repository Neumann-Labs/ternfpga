// Minimal soft-float libm for the bare-metal firmware: this LiteX/picolibc build ships no
// libm (no exp/cos/sin/sqrt/pow/floor/lrint), but the BitNet glue needs them for RMSNorm /
// RoPE / softmax. Accuracy ~1e-7 (double Taylor/Newton); soft-float MUL/ADD come from libcompiler_rt.
//
// Functions are fw_-prefixed (NOT standard names) on purpose: defining global `floor`/etc.
// makes picolibc's tinystdio pull its heavier float-printf path, which crashes %d formatting
// on this target. The firmware maps cos->fw_cos etc. via #define right before the glue headers,
// so the standard math symbols stay undefined and integer printf is used. Reused by #30/#31.
#ifndef FW_MATHF_H
#define FW_MATHF_H

static double fw_floor(double x)
{
	double t = (double)(long long)x;
	return (t > x) ? t - 1.0 : t;
}

static long fw_lrint(double x)            // ties-to-even (matches numpy round / the glue)
{
	double f = fw_floor(x), d = x - f;
	long fi = (long)f;
	if (d < 0.5) return fi;
	if (d > 0.5) return fi + 1;
	return (fi & 1L) ? fi + 1 : fi;
}

static double fw_sqrt(double x)
{
	if (x <= 0.0) return 0.0;
	double y = x, py = 0.0;
	for (int i = 0; i < 40 && y != py; i++) { py = y; y = 0.5 * (y + x / y); }
	return y;
}

static double fw_exp(double x)            // 2^k * poly(r), x = k*ln2 + r
{
	if (x > 700.0) return 1e308;
	if (x < -700.0) return 0.0;
	const double LN2 = 0.6931471805599453, INVLN2 = 1.4426950408889634;
	long k = fw_lrint(x * INVLN2);
	double r = x - (double)k * LN2;
	double e = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6 + r * (1.0 / 24 +
	           r * (1.0 / 120 + r * (1.0 / 720))))));
	double s = 1.0, b = 2.0;
	long kk = k < 0 ? -k : k;
	while (kk) { if (kk & 1) s *= b; b *= b; kk >>= 1; }
	return k < 0 ? e / s : e * s;
}

static double fw_log(double x)            // k*ln2 + log(m), m in [1,2)
{
	if (x <= 0.0) return -1e308;
	int k = 0;
	while (x >= 2.0) { x *= 0.5; k++; }
	while (x < 1.0) { x *= 2.0; k--; }
	double t = (x - 1.0) / (x + 1.0), t2 = t * t;
	double s = t * (2.0 + t2 * (2.0 / 3 + t2 * (2.0 / 5 + t2 * (2.0 / 7 + t2 * (2.0 / 9)))));
	return (double)k * 0.6931471805599453 + s;
}

static double fw_pow(double b, double e) { return fw_exp(e * fw_log(b)); }

static double fw_sin_core(double x)
{
	double x2 = x * x;
	return x * (1 + x2 * (-1.0 / 6 + x2 * (1.0 / 120 + x2 * (-1.0 / 5040 + x2 * (1.0 / 362880)))));
}

static double fw_sin(double x)
{
	const double PI = 3.141592653589793, TWOPI = 6.283185307179586, HP = 1.5707963267948966;
	x -= fw_floor(x / TWOPI + 0.5) * TWOPI;
	if (x > HP) x = PI - x;
	else if (x < -HP) x = -PI - x;
	return fw_sin_core(x);
}

static double fw_cos(double x) { return fw_sin(x + 1.5707963267948966); }

// Firmware maps the standard math names used by the glue to the fw_ implementations.
#define floor fw_floor
#define lrint fw_lrint
#define sqrt  fw_sqrt
#define exp   fw_exp
#define log   fw_log
#define pow   fw_pow
#define sin   fw_sin
#define cos   fw_cos

#endif /* FW_MATHF_H */

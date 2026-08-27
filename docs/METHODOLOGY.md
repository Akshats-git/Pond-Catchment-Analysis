# Methodology

How a contour map becomes a catchment area, and what the evidence is that the number
is right.

Every figure below is produced by the test suite, not transcribed into it. Regenerate
with `pytest tests/test_catchment_analytic.py tests/test_massbalance.py`.

---

## 1. The calculation

Six steps. Each one is a sentence.

### Step 1 — Contours to elevation points
Every vertex of every contour line becomes `(x, y, z)`, where `z` is the contour's label.

> *A contour line is a line of constant height, so every point on it has a known height.*

The height is rarely where you expect it in the file, so it is found by a **cascade** of
four strategies — 3D coordinates, an `<ExtendedData>` field, the placemark name, the
enclosing folder name — and the first one that explains at least 90% of the contours
wins. The provided sample resolves on the placemark name.

That coverage threshold is not decoration. The sample sheet contains one stray `land`
boundary polygon whose coordinates are 3D at `z = 30`; without the threshold the
z-coordinate strategy would win on that single feature and every elevation in the map
would be 30 m.

### Step 2 — Points to a grid (DEM)
Delaunay-triangulate the points; interpolate linearly onto a square metric grid.

The resolution is **derived from the data**. Contour lines of total length `L` spaced `w`
apart fill an area `A = L · w`, so

```
mean contour spacing = mapped area / total contour length
                     = 8.307e6 m² / 568,356 m = 14.6 m
grid resolution      = spacing / 4 = 3.65 m
```

Four cells across the typical gap between contours resolves the interpolated slope
without pretending to know more than the source does.

> *Flow algorithms need a grid of heights, so I fill in the gaps between contour lines.*

### Step 3 — Remove the stair-steps
Contour interpolation produces flat bands between contour lines — the surface is a
staircase, not a hillside. A Gaussian of `σ = spacing / 8` removes them.

> *Interpolating between contours makes flat steps; a light smooth turns the staircase
> back into a slope.*

The Gaussian must be **NaN-aware in normalised form**: fill invalid cells with zero,
smooth, and divide by the smoothed validity mask. Filling with the mean instead leaves a
phantom contribution the denominator does not account for, and cells near the data edge
inflate — it produced a 357 m peak on a map whose true maximum is 298 m.

**This step is worth an order of magnitude of accuracy. See §3, Table 1.**

### Step 4 — Fill the pits
Priority-flood (Barnes, Lehman & Mulla 2014): flood inward from the edges of the *data*,
adding a tiny `ε` so water keeps moving across flats.

> *Water shouldn't get stuck in a hole that only exists because the data is imperfect.*

Seeded from the array border **and** every cell adjacent to no-data. On the sample, 2.5%
of the grid falls outside the contour hull, so a basin can drain off the mapped area
through an interior hole without ever touching row 0.

The `ε` is load-bearing. With it, the sample sheet has 308 outlets and a maximum flow
accumulation of 158,262 cells. Without it — on a surface full of flat interpolated bands
— it has **108,526 outlets and a maximum accumulation of 712**, because a flat cell has
no downhill neighbour and becomes its own outlet.

### Step 5 — Flow direction and accumulation (D8)
Each cell sends all its water to the steepest of its eight neighbours:

```
S_i = (z_c − z_i) / d_i        d_i = res (4 sides), res·√2 (4 diagonals)
receiver(c) = argmax_i S_i
```

Dividing by the diagonal distance is what stops the routing drifting diagonally: a
diagonal neighbour is 41% further away, so an equal drop across it is a shallower slope.

Because every receiver is *strictly* lower than its donor, the flow graph is provably
acyclic and sorting cells by descending elevation is a topological order. One pass down
that order, adding each cell's running total to its receiver, gives the flow accumulation.

> *Every cell gives its water to its steepest neighbour; counting how much arrives tells
> me where the streams are.*

### Step 6 — Catchment = walk upstream
D8 gives each cell exactly one receiver, so the flow field is a *forest*. Following the
pointers backwards from the outlet collects its basin — and collects every cell exactly
once, so the traversal needs no visited-set.

```
A_cell      = res² · cos(φ_cell) / cos(φ₀)      ← latitude weighting
A_catchment = Σ A_cell   over the upstream mask
```

> *The catchment is every cell whose water eventually flows through the pond.*

---

## 2. What the answer is measured against

Three diagnostics travel with every catchment, because an area on its own is not a
result.

**Snap distance.** A requested point rarely lands on the routed channel, so the outlet is
snapped to the largest accumulation within `3 × contour spacing`. The radius scales with
the data because the channel itself moves — by about 90 m between the ensemble's grids —
and a fixed 30 m radius would snap to a different stream on each. The distance moved is
reported, since the answer is not for the point that was asked about.

**Edge contact.** The fraction of the catchment's perimeter facing *no-data or the array
border*. Above 15%, the true catchment continues off the map and the reported area is a
lower bound rather than a measurement. Testing the array border alone is not enough: a
catchment can leave the mapped area through an interior hole without touching row 0, and
doing so wrongly labelled the 395 ha basin complete.

**The ensemble.** The same site is delineated on 5.0, 3.5 and 2.5 m grids, each snapping
independently. Agreement means the answer is a property of the terrain; disagreement
means it is a property of the grid.

| Confidence | Coefficient of variation across grids |
|---|---|
| high | ≤ 0.10 |
| medium | ≤ 0.30 |
| low | > 0.30 |

---

## 3. Validation

### Test A — a valley with a provable answer

The surface is

```
z = 0.05·|x| + 0.01·y        over x ∈ [−500, 500], y ∈ [0, 1000]
```

a V-shaped valley draining north to south, with walls twenty times steeper than the floor.

From any cell the slope towards the channel is 0.05 and the slope down-valley is 0.01.
The diagonal between them drops `(0.05 + 0.01)/√2 = 0.042` per metre — *less* than 0.05 —
so D8 always takes the pure cross-valley step. Water reaches `x = 0` at the same `y` it
started from, then turns down the channel. Therefore the catchment of the channel point at
`y = Y` is everything above it:

```
A(Y) = 1000 · (1000 − Y)   m²
```

The valley is written out as a contour KML and read back through the **identical
pipeline** — parser, projection, interpolation, smoothing, fill, D8, accumulation,
delineation. It is placed 1,800 km from the sample sheet, so any coordinate that had
leaked into the code would show up as a visibly wrong answer rather than a plausible one.

#### Table 1 — error against the provable answer

| Configuration | Y=250 | Y=500 | Y=750 | worst missing cells |
|---|---|---|---|---|
| No smoothing, 5 m grid | −9.31% | −7.00% | **−27.93%** | 2,793 |
| No smoothing, 10 m grid | −5.99% | −6.00% | −17.96% | 449 |
| **σ = 2.46 m, 5 m grid** | **−0.66%** | **−0.00%** | **−1.99%** | **199** |
| σ = 2.46 m, 10 m grid | −4.65% | −4.00% | −13.96% | 349 |

Smoothing reduces the worst case from 2,793 missing cells to 199 — a factor of fourteen.

The 10 m row is the argument for deriving the resolution from the data rather than
accepting one. σ is tied to the contour spacing, so it is fixed in metres: 2.46 m is half
a cell at 5 m but a quarter of a cell at 10 m, and a quarter-cell Gaussian barely touches
the stair-steps.

#### What the residual 199 cells are

Not a routing error. The same 199 cells go missing at every Y, and the percentage moves
only because the catchment shrinks:

| Outlet at | channel elevation | on a contour line? | error (5 m grid) | missing cells |
|---|---|---|---|---|
| Y = 200 m | 2.00 m | **yes** | −0.00% | **0** |
| Y = 250 m | 2.50 m | no | −0.66% | 199 |
| Y = 300 m | 3.00 m | **yes** | −0.00% | **0** |
| Y = 450 m | 4.50 m | no | −0.91% | 199 |
| Y = 500 m | 5.00 m | **yes** | −0.00% | **0** |
| Y = 750 m | 7.50 m | no | −1.99% | 199 |
| Y = 800 m | 8.00 m | **yes** | −0.00% | **0** |

**Whenever the outlet cell coincides with a contour vertex the answer is exact — all
30,000 cells.** Where it falls between contours, the outlet's own row is lost: 199 cells,
being the 200-cell row minus the outlet itself.

The mechanism is **flat triangles**, a known hazard of contour interpolation. Near a
contour's V-apex the tent's two legs wrap around the valley axis, so a Delaunay triangle
can have all three vertices on the *same* contour — and a triangle whose corners are all
at one elevation interpolates to a plane. Down the channel of the raw interpolation there
are runs of bit-identical elevations several cells long; the smoothing shortens them but
cannot abolish them, so the outlet's exact position along the band stays uncertain by
about one cell.

Every error observed is a *missing* row, never a spurious one. A catchment that claimed
ground draining elsewhere would be the dangerous failure — it would promise a pond more
water than the terrain delivers.

### Test B — mass balance on the real map

Every cell drains to exactly one outlet, so the basins tile the map and their areas must
total the mapped area.

```
sum of all 308 basin areas = 8.309123 km²
mapped area                = 8.309123 km²      difference: 0.00000000%
```

Holds at 5.0, 3.5 and 2.5 m, on the analytic valley, and on every structural variant. It
is checked two ways that share no code: walking *downstream* to partition the map into
basins, and delineating every outlet *upstream* and confirming each cell is claimed
exactly once.

The check is cheap and unusually strict — a cycle in the flow graph, a cell draining into
no-data, an off-by-one in the neighbour offsets, or a catchment double-counting a cell all
break the sum.

### Test C — the resolution ensemble

| # | Outlet (lon, lat) | 5 m grid | ensemble | edge contact | relief | verdict |
|---|---|---|---|---|---|---|
| 1 | 81.286465, 21.240094 | 395.7 ha | 421.7 ± 18.4 ha | 2.6% | 27.0 m | best — 48% of map |
| 2 | 81.293549, 21.263343 | 104.0 ha | 103.0 ± 1.7 ha | 12.3% | 14.4 m | good, partly clipped |
| 3 | 81.284248, 21.262484 | 37.3 ha | 37.3 ± 0.0 ha | 2.9% | 17.8 m | very stable |
| 4 | 81.297453, 21.240094 | 35.8 ha | **14.1 ± 15.4 ha** | 2.4% | 10.0 m | **rejected — unstable** |
| 5 | 81.312393, 21.259544 | 33.3 ha | 35.0 ± 1.2 ha | 5.1% | 11.0 m | good |

Site 4 is the reason the ensemble exists. It measures 35.8 ha on the 5 m grid and 1.7 and
4.9 ha on the other two. Reported as a single number it looks like a perfectly good 36 ha
catchment; the ensemble gives a coefficient of variation of 1.09 and the site is rejected.

### Test D — the sub-tile test

The sample sheet is clipped to each of its four quadrants and re-analysed from scratch.
Each quadrant derives its own grid resolution, finds its largest catchment *within that
quadrant*, and satisfies mass balance. The four quadrants give four different answers,
differing by more than a factor of two.

A result that came from a hard-coded coordinate, a memorised resolution or a cached site
would survive every other test here and fail this one.

### Test E — structural variants

Eleven synthetic KML/KMZ files — 3D coordinates, `ExtendedData`, folder names, Polygon
rings, MultiGeometry, no XML namespace, spaced coordinates, a duplicate `labels` folder, a
stray 3D polygon, a KMZ archive — are carried all the way to a delineated catchment.
Parsing a file is not the same as being able to analyse it.

---

## 4. Known limitations

**Pond water levels are quantised to the contour interval.** A depression fills until it
spills over its lowest rim, and on a contour-derived DEM that rim sits on a contour line.
On the sample, 90% of pools of 20 cells or more have a spill elevation within 0.1 m of a
whole metre, with a median offset of 2.6 mm. Individual cell depths are *not* quantised,
because a cell's own elevation is interpolated.

**Catchments touching the edge of the data are lower bounds,** and are flagged as such
above 15% edge contact.

**DEM accuracy is bounded by the contour spacing** — 14.6 m on this sheet. The
interpolation cannot recover detail the contours never recorded, and near contour extremes
it produces flat triangles (see Table 1's residual).

**The projection carries a −0.2% area bias.** The equirectangular frame uses mid-latitude
constants (111320, 110540); at 21°N that is −0.04% east-west and −0.16% north-south
against WGS-84 — about 0.8 ha on a 395 ha catchment. Twenty times smaller than the ±4%
the ensemble reports for the same catchment. Shape distortion, which is the part D8 could
notice, is 0.12%, since a uniform scale error cancels when comparing neighbours.

**Outlet placement is uncertain by about one cell** along a flat interpolated band, as
Table 1 shows.

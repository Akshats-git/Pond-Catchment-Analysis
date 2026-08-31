# Methodology

How a contour map becomes a catchment area, where the pond goes, and what the evidence is
that the numbers are right.

Every figure below is produced by the test suite and not transcribed into it. Regenerate
with `pytest tests/test_catchment_analytic.py tests/test_massbalance.py`.

---

## 1. The calculation

Six steps. Each one is a sentence.

### Step 1. Contours to height points
Every vertex of every contour line becomes `(x, y, z)`, where `z` is the contour's label.

> *A contour line is a line of constant height, so every point on it has a known height.*

The height is rarely where you expect it in the file. It is found by a **cascade** of four
strategies: 3D coordinates, an `<ExtendedData>` field, the placemark name, the enclosing
folder name. The first one that explains at least 90% of the contours wins. The provided
sample resolves on the placemark name.

That coverage threshold is not decoration. The sample sheet contains one stray `land`
boundary polygon whose coordinates are 3D at `z = 30`. Without the threshold the
z-coordinate strategy would win on that single feature and every height in the map would
be 30 m.

### Step 2. Points to a grid
Delaunay-triangulate the points, then interpolate linearly onto a square metric grid.

The resolution is **worked out from the data**. Contour lines of total length `L` spaced
`w` apart fill an area `A = L · w`, so

```
mean contour spacing = mapped area / total contour length
                     = 8.307e6 m² / 568,356 m = 14.6 m
grid resolution      = spacing / 4 = 3.65 m
```

Four cells across the typical gap between contours resolves the interpolated slope without
pretending to know more than the source does.

> *Flow algorithms need a grid of heights, so the gaps between contour lines get filled
> in.*

### Step 3. Take out the stair steps
Contour interpolation produces flat bands between the lines. The surface comes out as a
staircase and not a hillside. A Gaussian of `σ = spacing / 8` takes them out.

> *Interpolating between contours makes flat steps. A light smooth turns the staircase
> back into a slope.*

The Gaussian has to be **NaN-aware in normalised form**: fill invalid cells with zero,
smooth, and divide by the smoothed validity mask. Filling with the mean instead leaves a
phantom contribution the denominator does not account for, and cells near the data edge
inflate. It produced a 357 m peak on a map whose true maximum is 298 m.

**This step is worth an order of magnitude of accuracy. See §4, Table 1.**

### Step 4. Fill the pits
Priority-flood (Barnes, Lehman & Mulla 2014). Flood inward from the edges of the *data*,
adding a tiny `ε` so water keeps moving across flats.

> *Water should not get stuck in a hole that only exists because the data is imperfect.*

Seeded from the array border **and** from every cell next to no-data. On the sample, 2.5%
of the grid falls outside the contour hull, so a basin can drain off the mapped area
through an interior hole without ever touching row 0.

The `ε` is load-bearing. With it, the sample sheet has 308 outlets and a maximum flow
accumulation of 158,262 cells. Without it, on a surface full of flat interpolated bands,
it has **108,526 outlets and a maximum accumulation of 712**, because a flat cell has no
downhill neighbour and becomes its own outlet.

### Step 5. Flow direction and accumulation
Each cell sends all its water to the steepest of its eight neighbours:

```
S_i = (z_c − z_i) / d_i        d_i = res (4 sides), res·√2 (4 diagonals)
receiver(c) = argmax_i S_i
```

Dividing by the diagonal distance is what stops the routing drifting diagonally. A
diagonal neighbour is 41% further away, so an equal drop across it is a shallower slope.

Because every receiver is *strictly* lower than its donor, the flow graph is provably
acyclic and sorting cells by descending height is a topological order. One pass down that
order, adding each cell's running total to its receiver, gives the flow accumulation.

> *Every cell gives its water to its steepest neighbour. Counting how much arrives tells
> you where the streams are.*

### Step 6. Catchment by walking upstream
D8 gives each cell exactly one receiver, so the flow field is a forest. Following the
pointers backwards from the outlet collects its basin, and collects every cell exactly
once, so the traversal needs no visited set.

```
A_cell      = res² · cos(φ_cell) / cos(φ₀)      ← latitude weighting
A_catchment = Σ A_cell   over the upstream mask
```

> *The catchment is every cell whose water eventually flows through the pond.*

---

## 2. Where the pond goes

A pond is only as good as the water that reaches it, so the search starts from the
drainage network rather than from the shape of the ground.

1. **The stream network.** A cell is on a stream when at least 0.5% of the mapped area
   drains through it, which is 4.2 ha on this sheet. The threshold is absolute and worked
   out from the input, never a percentile of flow accumulation. Accumulation is so skewed
   that percentile ranking scored 0.7 ha hollows at 0.98 alongside a 320 ha valley.
2. **Buildable ground.** Local slope under 3%, and at least 30 m inside the edge of the
   data, so neither the pond nor its catchment sits half off the map.
3. **Clear of the watercourse.** See §3. This is the rule that keeps the answer out of the
   river.
4. **Rank by catchment area.** Biggest basin first.
5. **Suppress each pick's whole catchment.** Not a square window. A square window returned
   five points strung along one stream at 391, 361, 215, 202 and 179 ha, every one of them
   nested inside the first. Removing the entire upstream mask makes the alternatives
   independent sub-basins, which is the only way a list of five sites means five choices.

---

## 3. Keeping the pond out of the river

### The failure

Ranking purely by catchment area asks for the cell that the most water passes through. On
any sheet with a river across it, the answer is the river.

On the provided map that produced a top site at 21.241343 N, 81.286758 E: 418.8 ha of an
830.8 ha sheet, with its outlet standing in the Shivnath. Drawn on satellite imagery the
recommendation sat in open water. A 3 m structure there would have been a 2.4 million m³
impoundment across a live channel, which is a dam and not a village pond.

### The rule

Two conditions, both needed.

**A channel draining more than 150 ha is a watercourse.** Absolute hectares, never a share
of the sheet. A share would scale with whatever was uploaded, so the biggest channel on a
20 ha farm map would be called a river and the rule would refuse to site anything at all.
A watercourse is a watercourse at 150 ha whether the sheet around it is 200 ha or
200 km².

**A site must stand 3 m above the watercourse it drains into.** Height above nearest
drainage, measured along the flow path rather than as a straight line, because what
matters is how far the pond bed stands above the channel that would flood it. Three metres
is a pond depth of freeboard. Below that the bed sits inside the channel or its
floodplain, so the monsoon fills it with silt and takes the bund with it.

Both numbers are configuration, at `SitingConfig.trunk_drainage_area_ha` and
`SitingConfig.min_height_above_trunk_m`.

Two cases have no answer, and they are opposite ones. A sheet with no channel over 150 ha
maps no watercourse at all, so every cell gets `+inf` and the rule does nothing. That is
the right answer for a farm-scale map, and it is why the analytic valley of §4 is
unaffected. On a sheet that does map one, a cell whose water leaves the sheet before
reaching it gets `-inf` instead. Such a cell sits at the edge of the data with an unknown
channel just beyond it, which is not the same as standing clear of one, and on this sheet
those cells are exactly the strip along the near bank of the river.

### The evidence

The rule is checked against an independent source: the OpenStreetMap water layer, sampled
onto the same grid. OSM knows nothing about the terrain model, so it is a fair judge of
whether a candidate cell stands in water.

| | candidate cells | standing in OSM water |
|---|---|---|
| Stream and buildable only | 2,413 | 310 (12.8%) |
| Plus clear of the watercourse | 566 | **0 (0.0%)** |

The five sites the service now returns stand 19, 48, 162, 333 and 441 m from mapped water,
and every one of them is on land.

The trunk it excludes is narrow. At 150 ha exactly one line on this sheet crosses the
threshold, taking 431 cells of a 638,472-cell grid. That line is the Shivnath. The rule
removes a river, not the drainage network the ponds are meant to sit on.

### What it cannot do

Terrain says where water collects. It cannot say whether that spot is already a house, a
road or a field somebody owns. Site 4 of the current run sits inside the built-up part of
the village, and nothing in a contour map could have told the service that. Land cover has
to come from outside, and the `app/providers/` seam is where it would go.

---

## 4. What the answer is measured against

Three diagnostics travel with every catchment, because an area on its own is not a result.

**Snap distance.** A requested point rarely lands on the routed channel, so the outlet is
snapped to the largest accumulation within `3 × contour spacing`. The radius scales with
the data because the channel itself moves, by about 90 m between the ensemble's grids, and
a fixed 30 m radius would snap to a different stream on each. The distance moved is
reported, since the answer is not for the point that was asked about.

**Edge contact.** The fraction of the catchment's perimeter facing *no-data or the array
border*. Above 15%, the true catchment carries on off the map and the reported area is a
floor rather than a measurement. Testing the array border alone is not enough. A catchment
can leave the mapped area through an interior hole without touching row 0, and doing so
wrongly labelled a 395 ha basin complete.

**The ensemble.** The same site is traced on 5.0, 3.5 and 2.5 m grids, each snapping
independently. Agreement means the answer is a property of the terrain. Disagreement means
it is a property of the grid.

| Confidence | Coefficient of variation across grids |
|---|---|
| high | ≤ 0.10 |
| medium | ≤ 0.30 |
| low | > 0.30 |

---

## 5. Validation

### Test A. A valley with a provable answer

The surface is

```
z = 0.05·|x| + 0.01·y        over x ∈ [−500, 500], y ∈ [0, 1000]
```

a V-shaped valley draining north to south, with walls twenty times steeper than the floor.

From any cell the slope towards the channel is 0.05 and the slope down-valley is 0.01. The
diagonal between them drops `(0.05 + 0.01)/√2 = 0.042` per metre, which is *less* than
0.05, so D8 always takes the pure cross-valley step. Water reaches `x = 0` at the same `y`
it started from, then turns down the channel. So the catchment of the channel point at
`y = Y` is everything above it:

```
A(Y) = 1000 · (1000 − Y)   m²
```

The valley is written out as a contour KML and read back through the **identical
pipeline**: parser, projection, interpolation, smoothing, fill, D8, accumulation,
delineation. It is placed 1,800 km from the sample sheet, so any coordinate that had
leaked into the code would show up as a visibly wrong answer rather than a plausible one.

#### Table 1. Error against the provable answer

| Configuration | Y=250 | Y=500 | Y=750 | worst missing cells |
|---|---|---|---|---|
| No smoothing, 5 m grid | −9.31% | −7.00% | **−27.93%** | 2,793 |
| No smoothing, 10 m grid | −5.99% | −6.00% | −17.96% | 449 |
| **σ = 2.46 m, 5 m grid** | **−0.66%** | **−0.00%** | **−1.99%** | **199** |
| σ = 2.46 m, 10 m grid | −4.65% | −4.00% | −13.96% | 349 |

Smoothing brings the worst case down from 2,793 missing cells to 199, a factor of
fourteen.

The 10 m row is the argument for working the resolution out from the data rather than
accepting one. σ is tied to the contour spacing, so it is fixed in metres. 2.46 m is half
a cell at 5 m but a quarter of a cell at 10 m, and a quarter-cell Gaussian barely touches
the stair steps.

#### What the residual 199 cells are

Not a routing error. The same 199 cells go missing at every Y, and the percentage moves
only because the catchment shrinks:

| Outlet at | channel height | on a contour line? | error (5 m grid) | missing cells |
|---|---|---|---|---|
| Y = 200 m | 2.00 m | **yes** | −0.00% | **0** |
| Y = 250 m | 2.50 m | no | −0.66% | 199 |
| Y = 300 m | 3.00 m | **yes** | −0.00% | **0** |
| Y = 450 m | 4.50 m | no | −0.91% | 199 |
| Y = 500 m | 5.00 m | **yes** | −0.00% | **0** |
| Y = 750 m | 7.50 m | no | −1.99% | 199 |
| Y = 800 m | 8.00 m | **yes** | −0.00% | **0** |

**Whenever the outlet cell coincides with a contour vertex the answer is exact, all 30,000
cells.** Where it falls between contours, the outlet's own row is lost: 199 cells, being
the 200-cell row minus the outlet itself.

The mechanism is **flat triangles**, a known hazard of contour interpolation. Near a
contour's V-apex the tent's two legs wrap around the valley axis, so a Delaunay triangle
can have all three vertices on the *same* contour, and a triangle whose corners are all at
one height interpolates to a plane. Down the channel of the raw interpolation there are
runs of bit-identical heights several cells long. Smoothing shortens them but cannot
abolish them, so the outlet's exact position along the band stays uncertain by about one
cell.

Every error observed is a *missing* row, never a spurious one. A catchment that claimed
ground draining elsewhere would be the dangerous failure, because it would promise a pond
more water than the terrain delivers.

### Test B. Mass balance on the real map

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

The check is cheap and unusually strict. A cycle in the flow graph, a cell draining into
no-data, an off-by-one in the neighbour offsets, or a catchment double-counting a cell all
break the sum.

### Test C. The resolution ensemble

The five sites the service returns, traced on the 5 m grid and cross-checked on 3.5 and
2.5 m.

| # | Outlet (lon, lat) | 5 m grid | ensemble | edge contact | relief | above the channel | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 81.300200, 21.250588 | 119.7 ha | 120.3 ± 0.7 ha | 0.0% | 15.0 m | 3.0 m | best, and stable |
| 2 | 81.290079, 21.244029 | 37.5 ha | **63.7 ± 17.9 ha** | 2.3% | 16.0 m | 4.0 m | **medium, read the spread** |
| 3 | 81.288778, 21.248326 | 18.4 ha | 21.3 ± 2.0 ha | 0.0% | 10.0 m | 6.2 m | good |
| 4 | 81.298898, 21.248145 | 18.1 ha | 18.3 ± 0.1 ha | 0.0% | 6.1 m | 3.1 m | very stable |
| 5 | 81.292200, 21.247376 | 12.9 ha | 13.1 ± 1.2 ha | 0.0% | 11.0 m | 4.4 m | good |

Site 2 is why the ensemble exists. It measures 37.5 ha on the 5 m grid and averages
63.7 ha across the three, a coefficient of variation of 0.28. Reported as a single number
it looks like a perfectly good 38 ha catchment. Reported with its spread, a reader knows
to walk it before acting on it.

### Test D. The sub-tile test

The sample sheet is clipped to each of its four quadrants and re-analysed from scratch.
Each quadrant works out its own grid resolution, finds its largest catchment *within that
quadrant*, and satisfies mass balance. The four quadrants give four different answers,
differing by more than a factor of two.

A result that came from a hard-coded coordinate, a memorised resolution or a cached site
would survive every other test here and fail this one.

### Test E. Structural variants

Eleven synthetic KML and KMZ files are carried all the way to a delineated catchment: 3D
coordinates, `ExtendedData`, folder names, Polygon rings, MultiGeometry, no XML namespace,
spaced coordinates, a duplicate `labels` folder, a stray 3D polygon, a KMZ archive.
Parsing a file is not the same as being able to analyse it.

---

## 6. Water

**Runoff is SCS-CN applied per rain day and summed.** It is an event model. Applied to a
year's rain as one storm it returns a 92% runoff coefficient, which no catchment on earth
produces. Applied day by day and added up it returns 8 to 16% on this terrain, depending
on how the rain is spread across days. Both numbers are reported, the wrong one
deliberately, because a report that only shows the right answer cannot show why it is
right.

**Rainfall comes from ten years of Open-Meteo daily records** for the chosen site, which
is free and needs no key. The provider hands back every wet day of the ten and says how
many years that is; the runoff model divides by that at the end, which makes the reported
figure the mean of the ten annual runoffs. Averaging the years into one synthetic year
first would flatten the big days, and the big days are most of the runoff. If the service
cannot be reached, a documented regional climatology answers and the response says so.

**ERA5 is a reanalysis on a roughly 25 km grid.** It spreads a village's rain over more
days than a rain gauge in that village would record, and runoff is quadratic in daily
depth, so a flatter series yields less runoff. The same place reads 16% of rainfall as
runoff on the seeded climatology (1,200 mm over 55 days) and 8% on ten years of ERA5
(1,397 mm over 118 days). The observed figure is the more defensible starting point and
the more conservative one, but a design being costed should be checked against the nearest
gauge.

**The stage-storage curve is integrated from the grid, not assumed from a shape.** At each
stage the pond is every cell the water reaches from the outlet without crossing ground
that stands above it. Where one step multiplies the water surface, the pond has topped a
ridge, and the volume below that step is what the site really holds. The frustum formula a
spreadsheet would use is reported beside the integral as a cross-check, and underestimates
it by 26 to 65% across the sample's sites, because real ground widens as it rises faster
than straight sides do.

---

## 7. Known limitations

**Pond water levels are quantised to the contour interval.** A depression fills until it
spills over its lowest rim, and on a contour-derived grid that rim sits on a contour line.
On the sample, 90% of pools of 20 cells or more have a spill height within 0.1 m of a
whole metre, with a median offset of 2.6 mm. Individual cell depths are *not* quantised,
because a cell's own height is interpolated.

**Catchments touching the edge of the data are floors,** and are flagged as such above 15%
edge contact.

**Grid accuracy is bounded by the contour spacing,** 14.6 m on this sheet. The
interpolation cannot recover detail the contours never recorded, and near contour extremes
it produces flat triangles. See Table 1's residual.

**Land cover is not known.** The watercourse rule keeps a site out of the river. Nothing
here keeps it out of a village, a road or somebody's field. See §3.

**The projection carries a −0.2% area bias.** The equirectangular frame uses mid-latitude
constants (111320, 110540). At 21°N that is −0.04% east-west and −0.16% north-south
against WGS-84, about 0.8 ha on a 395 ha catchment, twenty times smaller than the ±4% the
ensemble reports for the same catchment. Shape distortion, which is the part D8 could
notice, is 0.12%, since a uniform scale error cancels when comparing neighbours.

**Outlet placement is uncertain by about one cell** along a flat interpolated band, as
Table 1 shows.

**The usable capacity is not monotone in the requested depth.** The stage-storage curve
has a fixed twelve steps between the bed and the target, so a deeper target means coarser
steps and the spill stage lands somewhere else. That is a property of the curve and not of
the ground. The capacity at the requested depth is monotone, and is the figure to compare.

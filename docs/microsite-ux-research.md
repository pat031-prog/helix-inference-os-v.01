# Microsite UX Research

## Goal

Make the public front page explain the project earlier, feel more singular, and hold up on mobile without collapsing into a generic card stack.

## What the reference images are telling us

The two references point to a very specific visual grammar:

- warm monochrome base instead of a glossy SaaS palette
- thick black outlines and hard offset shadows
- rounded modular widgets instead of flat sections
- dark "instrument" panels mixed with light paper panels
- segmented controls, dense information tiles, and a dashboard rhythm
- playful but controlled contrast between editorial text and utility metrics

This is not pure brutalism and not classic finance UI. The closest description is a soft neobrutalist control-room style with editorial data storytelling layered on top.

## Research inputs

### Homepage clarity and first-screen priorities

- Baymard notes that important recurring actions should be visible in the initial viewport, above the fold, instead of being pushed down by banners or oversized visuals:
  [Grocery and Food Delivery Site UX: Allow Users to Add "Past Purchases" to the Cart from the Homepage](https://baymard.com/blog/grocery-food-delivery-orders)
- Baymard also shows that homepage wording must carry the full scope of a destination on mobile, because users lose context easily when labels are too short:
  [Always Provide the Full Scope for Links on Mobile Homepages](https://baymard.com/blog/mobile-homepage-provide-full-scope)
- Baymard further argues that users infer what a site is from what the homepage surfaces first, and that a narrow presentation causes wrong assumptions about the product range:
  [Homepage UX: Featuring Product Breadth](https://baymard.com/blog/inferring-product-catalog-from-homepage)

Design implication for Helix:

- the front page cannot be a thin index anymore
- it has to explain the thesis on first view
- it has to expose the strongest verified and frontier results directly
- route labels must explain what each destination is, not just name it

### Mobile-first data visualization

- NN/g recommends that decorative mobile imagery should be avoided unless it adds informational value:
  [Images on Mobile](https://www.nngroup.com/videos/mobile-images/?lm=supporting-multiple-location-users&pt=article)
- Datawrapper explicitly recommends designing with mobile in mind during the visualization process, checking responsive previews frequently, and in some cases designing in mobile view first:
  [How to move around and set the size of your locator map](https://academy.datawrapper.de/article/155-how-to-move-and-set-the-size-of-a-locator-map)
- Datawrapper explains that axis labels often waste precious horizontal room on narrow screens, and that titles and descriptions should carry more of the explanatory load:
  [Why many Datawrapper charts don't include axis labels](https://academy.datawrapper.de/article/239-why-datawrapper-does-not-include-axis-labels-for-many-charts)
- Datawrapper also recommends narrow aspect ratios, shorter labels, keys, and layout changes specifically for mobile:
  [How to make locator maps look good on mobile devices](https://academy.datawrapper.de/article/338-locator-maps-for-mobile-devices)

Design implication for Helix:

- charts on mobile should lose nonessential labels before they lose readability
- titles, subtitles, and nearby explanatory copy should carry meaning
- first-screen SVGs must be compact, high-contrast, and understandable without tiny axis text
- the mobile layout should preserve widget identity, not just collapse into a long generic list

## Translation into the current site

### Front page

- move the hybrid frontier and the strongest metrics directly to `/`
- explain "Transformer memory" vs "Hybrid memory" in plain language
- show one first-screen visual that explains the thesis, not just the navigation
- keep deep links to `/research`, `/frontier`, and `/app`, but label them by purpose

### Visual identity

- keep the warm paper background and black outlines
- introduce more dark metric panels to echo the reference widgets
- use rounded tiles and inset subcards to create density without clutter
- make the front page feel like a lab dashboard, not a generic hero + cards landing

### Mobile behavior

- reduce hard shadow size and padding on small screens
- switch visual sections to one column earlier
- keep nav pills horizontally scrollable instead of wrapping into messy rows
- preserve large, readable numeric emphasis in SVGs

## V2 direction: compact instrument panel

The next iteration should not simply "shrink the cards". The stronger move is to reorganize the site around a tighter control-panel rhythm:

- first viewport must combine thesis, frontier visual, and proof strip in one composition
- strongest hybrid result belongs in the hero, not lower in the page
- explanatory copy should become shorter and more layered: thesis first, details second
- the visual system should alternate dark metric panels with light paper panels to feel closer to a device UI
- SVGs should behave like dashboard widgets, with mobile-specific layouts instead of just scaled-down desktop charts

The reference images reinforce a few concrete traits that are worth keeping:

- compact rows of modules with different visual weights instead of repeated equal cards
- narrow shadows and slightly smaller radii for a more refined, less toy-like result
- metric tiles that look like displays or instrument readouts
- labels outside the most compressed chart areas whenever mobile space gets tight
- section density high enough that the page feels editorial and technical, not airy or marketing-led

## Next iteration ideas

- add one mobile-only SVG layout variant for the hybrid frontier panel
- tighten typographic contrast further with a more distinctive display face if a webfont is acceptable later
- introduce one or two more visual signatures from the references, such as dark segmented status tiles or inverted mini-panels inside the hero
- test the home and microsite visually on a real narrow viewport after each major SVG change

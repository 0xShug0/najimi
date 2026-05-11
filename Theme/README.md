# Najimi Themes

This folder contains theme definitions used to shape the visual tone of Najimi.

Current theme files are simple JSON documents with:

- an `id`
- a `displayName`
- a `version`
- a `colors` object

The color set covers both shared accent colors and UI surfaces for light and dark appearance modes. In the current example, that includes:

- brand colors such as `primary`, `secondary`, and `accent`
- text colors such as `text`, `textSecondary`, and `textTertiary`
- surfaces such as `background`, `surface`, `surfaceMuted`, and `surfaceElevated`
- borders, status colors, gradients, and shadow tokens

## Included theme

- [`najimi-theme-cyber-idol.json`](./najimi-theme-cyber-idol.json): a bright pink-and-cyan theme with a more playful idol-inspired look.

## Format notes

Most colors are defined as hex values:

```json
{ "hex": "#FF2AA6" }
```

Some tokens also include opacity:

```json
{ "hex": "#000000", "opacity": 0.46 }
```

Tokens that need separate light and dark values are grouped like this:

```json
{
  "light": { "hex": "#FFF8FC" },
  "dark": { "hex": "#090812" }
}
```

## Theme customization

Najimi will support customizable themes more directly over time.

The goal is to let people do more than just pick a built-in look. Theme customization is intended to grow into a supported part of the companion experience, including the ability to import user-created themes so the app’s color mood and presentation can feel more personal.

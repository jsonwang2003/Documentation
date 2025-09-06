# Semantic Timeline Classes Test

Testing the Obsidian plugin semantic classes in Quartz:

## Dot Style Timeline

```timeline
[dot]
+ 2024
+ Dot Style
+ This uses the semantic dot class from Obsidian plugin

+ 2025  
+ Another Event
+ With outlined dots and clean styling
```

## Success Timeline

```timeline
[success]
+ Q1 2024
+ Project Success
+ Using the success semantic class with green styling

+ Q2 2024
+ Milestone Achieved
+ Green dots indicate successful completion
```

## Warning Timeline

```timeline
[warning]
+ Phase 1
+ Important Notice
+ Warning style with dashed connector line

+ Phase 2
+ Caution Required
+ Double-ring dot pattern for warnings
```

## Error Timeline

```timeline
[error]
+ 09:15 AM
+ System Alert
+ Error style with large red dots

+ 09:30 AM
+ Issue Resolved
+ Diamond-shaped inner element
```

## Card Content Style

```timeline
[success, card]
+ 2024
+ Card Style Title
+ This content is displayed in a card format with elevated styling and rounded corners.

+ 2025
+ Another Card
+ Cards provide clear visual separation between timeline items.
```

## Elevated Content Style

```timeline
[warning, elevated]
+ Phase 1
+ Elevated Style
+ The elevated style features an arrow-shaped title background.

+ Phase 2
+ Continued Progress
+ Creates a distinctive visual hierarchy for important content.
```

## Bordered Content Style

```timeline
[error, bordered]
+ Critical Event
+ Bordered Style
+ Bordered content has a colored left border matching the timeline dot.

+ Follow-up
+ Resolution
+ Clean and minimal styling with clear content boundaries.
```

## Mixed Combinations

```timeline
[dot, card]
+ 2024
+ Dot with Card
+ Combining dot style with card content layout

[success, elevated]
+ 2024
+ Success with Elevation
+ Green success dots with elevated content style

[warning, bordered]  
+ 2024
+ Warning with Border
+ Warning styling with bordered content layout
```

## Legacy Class Support

```timeline
[line-2, body-1]
+ 2024
+ Legacy Classes
+ Old syntax still works for backward compatibility

[line-3, body-2]
+ 2024  
+ Another Legacy
+ Maps to success + elevated automatically
```

These semantic classes now work identically in both Quartz and Obsidian!

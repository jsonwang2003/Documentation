# Material UI Timeline Demo

This page demonstrates the Material UI inspired timeline component natively supported in Quartz.

## Basic Material Timeline

Here's a simple timeline using Material UI styling:

```timeline
+ pre 17th century
+ Origins of coffee
+ The modern version of roasted coffee originated in Arabia. During the 13th century, coffee was extremely popular with the Muslim community for its stimulant powers, which proved useful during long prayer sessions. By parching and boiling the coffee beans, rendering them infertile, the Arabs were able to corner the market on coffee crops.

+ 17th century
+ Europe and coffee
+ In 1616, the Dutch founded the first European-owned coffee estate in Sri Lanka, then Ceylon, then Java in 1696. The French began growing coffee in the Caribbean, followed by the Spanish in Central America and the Portuguese in Brazil. European coffee houses sprang up in Italy and later France, where they reached a new level of popularity.

+ 18th century
+ Coffee comes to America
+ Coffee first came to America in the early 1700s, though tea remained more popular until the Boston Tea Party of 1773 made coffee the patriotic drink of choice. Coffee houses became important meeting places for revolutionaries and businessmen alike.
```

## Dense Timeline

A more compact version for space-efficient layouts:

```timeline
[dense]
+ Q1 2023
+ Planning Phase
+ Initial project planning and requirements gathering. Team formation and resource allocation.

+ Q2 2023
+ Development Start
+ Core development begins with architecture design and initial prototyping.

+ Q3 2023
+ Beta Testing
+ Closed beta launch with selected users and partners for feedback collection.

+ Q4 2023
+ Public Launch
+ Full product release with comprehensive marketing campaign and user onboarding.
```

## Right-Aligned Timeline

All content on the left side:

```timeline
[right]
+ January 2024
+ Project Kickoff
+ Project officially started with stakeholder meeting and goal setting.

+ March 2024
+ MVP Development
+ Minimum viable product development phase with core features implementation.

+ June 2024
+ User Testing
+ Comprehensive user testing phase with iterative improvements and bug fixes.

+ September 2024
+ Production Release
+ Final production release with full feature set and documentation.
```

## Success Timeline

Timeline with success theme for achievements:

```timeline
[success]
+ 2020
+ Company Founded
+ Started the journey with a small team and big dreams in a garage.

+ 2021
+ First Product Launch
+ Successfully launched our first product to positive market reception.

+ 2022
+ Series A Funding
+ Secured significant funding to accelerate growth and expansion plans.

+ 2023
+ Market Leader
+ Achieved market leadership position with innovative solutions and customer focus.
```

## Large Timeline

Bigger timeline for important milestones:

```timeline-labeled
[dot, large, primary]
date: 2019
title: Research & Development
content: Extensive research phase focusing on market analysis, technology stack selection, and competitive landscape evaluation. This foundational period set the stage for all future development efforts.

date: 2020-2021
title: Product Development
content: Intensive development period where the core product was built from the ground up. Multiple iterations, user feedback sessions, and continuous improvement cycles were implemented.

date: 2022
title: Market Entry
content: Strategic market entry with targeted marketing campaigns, partnership development, and customer acquisition initiatives. Focus on building brand recognition and market presence.

date: 2023-2024
title: Scale & Growth
content: Rapid scaling phase with team expansion, infrastructure development, and international market penetration. Implementation of advanced features and enterprise solutions.
```

## Timeline Style Variations

### Warning Timeline
```timeline
[warning, dense]
+ Step 1
+ Initial Setup
+ Configure your development environment and install required dependencies.

+ Step 2
+ Implementation
+ Follow the implementation guidelines carefully to avoid common pitfalls.

+ Step 3
+ Testing
+ Thoroughly test your implementation before deploying to production.
```

### Error Timeline
```timeline
[error, small]
+ Issue Detected
+ System Alert
+ Critical system error detected in production environment.

+ Investigation
+ Root Cause Analysis
+ Team investigating the issue and identifying potential causes.

+ Resolution
+ Fix Deployed
+ Hotfix deployed and systems restored to normal operation.
```

## Features

This Material UI inspired timeline implementation provides:

- **Clean, modern design** following Material Design principles
- **Alternating layout** for better visual flow and readability
- **Card-based content** with subtle shadows and hover effects
- **Responsive design** that adapts to mobile devices
- **Multiple style variants**:
  - `dense` - Compact spacing for more content
  - `right` - All content on left side
  - `alternate` - All content on right side
  - `large` / `small` - Size variations
  - `primary` / `success` / `warning` / `error` - Color themes
- **Smooth animations** with Material Design easing curves
- **Interactive elements** with hover effects and transitions
- **Dark theme support** with automatic color adaptation

## Usage

To use Material UI timelines in your Quartz site:

1. Use `timeline` or `timeline-labeled` code blocks
2. Add optional Material UI style classes: `[dense, success, large]`
3. For `timeline`: Use `+` at the start of each line (time, title, content)
4. For `timeline-labeled`: Use `date:`, `title:`, and `content:` labels

The timeline automatically creates Material UI style cards with proper spacing, shadows, and alternating layout for an elegant, professional appearance!

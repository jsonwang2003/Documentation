import { QuartzComponent, QuartzComponentConstructor } from "./types"

// Timeline component scripts
// @ts-ignore
import timelineScript from "./scripts/timeline.inline"

const TimelineScript: QuartzComponent = () => {
  return null // This component doesn't render anything directly
}

TimelineScript.afterDOMLoaded = timelineScript

export default (() => TimelineScript) satisfies QuartzComponentConstructor

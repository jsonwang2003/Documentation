import { QuartzComponent, QuartzComponentConstructor } from "./types"

// Timeline component styles and scripts
import timelineStyle from "./styles/timeline-combined.scss"
// @ts-ignore
import timelineScript from "./scripts/timeline.inline"

const Timeline: QuartzComponent = () => {
  // This component ensures timeline styles and scripts are loaded
  // The actual timeline rendering is handled by the timeline transformer
  return null
}

Timeline.css = timelineStyle
Timeline.beforeDOMLoaded = timelineScript

export default (() => Timeline) satisfies QuartzComponentConstructor

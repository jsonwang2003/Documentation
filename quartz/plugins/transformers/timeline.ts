import { QuartzTransformerPlugin } from "../types"
import { Root, Element } from "hast"
import { visit } from "unist-util-visit"

export interface Options {
  enableTimeline: boolean
  enableTimelineLabeled: boolean
}

const defaultOptions: Options = {
  enableTimeline: true,
  enableTimelineLabeled: true,
}

// Helper function to escape HTML
function escapeHtml(text: string): string {
  const map: { [key: string]: string } = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }
  return text.replace(/[&<>"']/g, (char) => map[char])
}

// Regex for extracting class options like [dense, success]
const classRegex = /^\[([^\]]+)\]/

// Convert class string to array
function toClassArray(classString: string): string[] {
  return classString
    .replace(/^\[|\]$/g, '')
    .split(',')
    .map(cls => cls.trim())
    .filter(cls => cls.length > 0)
}

// Parse timeline content (+ format)
function parseTimelineContent(content: string): Array<{time: string, title: string, description: string}> {
  const lines = content.split('\n').filter(line => line.trim())
  const events: Array<{time: string, title: string, description: string}> = []
  
  for (let i = 0; i < lines.length; i += 3) {
    if (lines[i]?.startsWith('+ ') && lines[i + 1]?.startsWith('+ ') && lines[i + 2]?.startsWith('+ ')) {
      events.push({
        time: lines[i].substring(2).trim(),
        title: lines[i + 1].substring(2).trim(),
        description: lines[i + 2].substring(2).trim()
      })
    }
  }
  
  return events
}

// Parse timeline-labeled content
function parseTimelineLabeledContent(content: string): Array<{time: string, title: string, description: string}> {
  const sections = content.split(/(?=^date:)/m).filter(section => section.trim())
  const events: Array<{time: string, title: string, description: string}> = []
  
  for (const section of sections) {
    const lines = section.trim().split('\n')
    let time = '', title = '', description = ''
    
    for (const line of lines) {
      if (line.startsWith('date:')) {
        time = line.substring(5).trim()
      } else if (line.startsWith('title:')) {
        title = line.substring(6).trim()
      } else if (line.startsWith('content:')) {
        description = line.substring(8).trim()
      }
    }
    
    if (time && title && description) {
      events.push({ time, title, description })
    }
  }
  
  return events
}

export const Timeline: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "Timeline",
    htmlPlugins() {
      return [
        () => {
          return (tree: Root, file) => {
            visit(tree, "element", (node: Element, index, parent) => {
              if (node.tagName === "pre") {
                const codeElement = node.children.find(child => 
                  child.type === "element" && child.tagName === "code"
                ) as Element | undefined
                
                if (codeElement && codeElement.properties) {
                  const className = codeElement.properties.className as string[] | undefined
                  const lang = className?.[0]?.replace("language-", "")
                  
                  if (
                    (opts.enableTimeline && lang === "timeline") ||
                    (opts.enableTimelineLabeled && lang === "timeline-labeled")
                  ) {
                    const source = codeElement.children[0]?.type === "text" 
                      ? codeElement.children[0].value as string 
                      : ""
                    const isLabeled = lang === "timeline-labeled"
                    
                    // Extract classes from the first line if present
                    const classMatch = source.match(classRegex)
                    let content = source
                    let additionalClasses: string[] = []
                    
                    if (classMatch) {
                      additionalClasses = toClassArray(classMatch[0])
                      content = source.replace(classRegex, "").trim()
                    }
                    
                    // Parse the timeline content
                    const events = isLabeled 
                      ? parseTimelineLabeledContent(content)
                      : parseTimelineContent(content)
                    
                    // Create timeline container with proper classes
                    const timelineClasses = ["timeline", ...additionalClasses].join(" ")
                    
                    // Convert to Material UI Timeline structure
                    const timelineHtml = `<div class="${timelineClasses}">
                    ${events.map((event, index) => 
                    `  <div class="timeline-item" data-index="${index}">
                        <div class="timeline-time">${escapeHtml(event.time)}</div>
                        <div class="timeline-separator">
                          <div class="timeline-dot"></div>
                          <div class="timeline-connector"></div>
                        </div>
                        <div class="timeline-content">
                          <div class="timeline-title">${escapeHtml(event.title)}</div>
                          <div class="timeline-description">${escapeHtml(event.description)}</div>
                        </div>
                      </div>`).join('\n')}
                    </div>`

                    // Replace the pre element with timeline HTML
                    if (parent && typeof index === 'number') {
                      parent.children[index] = {
                        type: "raw",
                        value: timelineHtml
                      } as any
                    }

                    // Mark file as having timeline
                    file.data.hasTimeline = true
                  }
                }
              }
            })
          }
        }
      ]
    },
  }
}

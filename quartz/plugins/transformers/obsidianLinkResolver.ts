import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"
import path from "path"

interface FileIndex {
  [filename: string]: string[] // filename -> array of full paths
}

export const ObsidianLinkResolver: QuartzTransformerPlugin = () => {
  return {
    name: "ObsidianLinkResolver",
    markdownPlugins(ctx) {
      const fileIndex: FileIndex = {}
      
      // Build index of all files
      const buildFileIndex = () => {
        if (ctx?.allSlugs) {
          for (const slug of ctx.allSlugs) {
            const filename = path.basename(slug)
            const filenameWithoutExt = path.basename(slug, path.extname(slug))
            
            // Index both with and without extension
            if (!fileIndex[filename]) fileIndex[filename] = []
            if (!fileIndex[filenameWithoutExt]) fileIndex[filenameWithoutExt] = []
            
            fileIndex[filename].push(slug)
            fileIndex[filenameWithoutExt].push(slug)
          }
        }
      }

      const resolveWikilink = (linkText: string, currentFile: string): string => {
        // Remove display text if present (e.g., "File|Display Text" -> "File")
        const [targetFile] = linkText.split("|")
        const cleanTarget = targetFile.trim()
        
        // Get current directory
        const currentDir = path.dirname(currentFile)
        
        // If it contains a slash, it's a relative path
        if (cleanTarget.includes("/")) {
          // For relative paths like "String/index", resolve relative to current directory
          const relativePath = currentDir ? `${currentDir}/${cleanTarget}` : cleanTarget
          
          console.log(`Resolving wikilink: [[${cleanTarget}]] from ${currentFile}`)
          console.log(`Current dir: ${currentDir}`)
          console.log(`Trying relative path: ${relativePath}`)
          
          // Check if this relative path exists in our file index
          const relativeExists = Object.values(fileIndex).some(paths => 
            paths.some(p => p === relativePath || p === relativePath.replace(/\/index$/, ""))
          )
          
          if (relativeExists) {
            console.log(`Found relative path: ${relativePath}`)
            return relativePath
          }
          
          // If relative path doesn't exist, try as absolute path
          const absoluteExists = Object.values(fileIndex).some(paths => 
            paths.some(p => p === cleanTarget || p === cleanTarget.replace(/\/index$/, ""))
          )
          
          if (absoluteExists) {
            console.log(`Found absolute path: ${cleanTarget}`)
            return cleanTarget
          }
          
          console.log(`No match found, returning relative path: ${relativePath}`)
          return relativePath
        }
        
        // Look up in file index for simple names
        const possiblePaths = fileIndex[cleanTarget] || []
        
        if (possiblePaths.length === 0) {
          console.warn(`Could not resolve wikilink: [[${cleanTarget}]]`)
          return cleanTarget
        }
        
        if (possiblePaths.length === 1) {
          return possiblePaths[0]
        }
        
        // Multiple matches - try to find the best one
        // Prefer files in the same directory as current file
        const sameDir = possiblePaths.find(p => path.dirname(p) === currentDir)
        if (sameDir) return sameDir
        
        // Return the shortest path
        return possiblePaths.reduce((shortest, current) => 
          current.length < shortest.length ? current : shortest
        )
      }

      return [
        () => {
          return (tree: Root, file) => {
            buildFileIndex()
            const currentFile = file.data.slug as string || ""
            
            // Process wikilinks in text nodes
            visit(tree, "text", (node: any, index, parent) => {
              if (!node.value || typeof node.value !== 'string') return
              
              // Find all wikilinks in the text
              const wikilinkPattern = /\[\[([^\]]+)\]\]/g
              const text = node.value
              let match
              const replacements: Array<{start: number, end: number, replacement: any}> = []
              
              while ((match = wikilinkPattern.exec(text)) !== null) {
                const fullMatch = match[0]
                const linkContent = match[1]
                const [target, displayText] = linkContent.split('|')
                
                const resolvedPath = resolveWikilink(linkContent, currentFile)
                
                // Create a link node
                const linkNode = {
                  type: 'link',
                  url: resolvedPath,
                  children: [{
                    type: 'text',
                    value: displayText || target.trim()
                  }]
                }
                
                replacements.push({
                  start: match.index,
                  end: match.index + fullMatch.length,
                  replacement: linkNode
                })
              }
              
              // Apply replacements in reverse order to maintain indices
              if (replacements.length > 0 && parent && typeof index === 'number') {
                const newNodes = []
                let lastEnd = 0
                
                for (const replacement of replacements) {
                  // Add text before the wikilink
                  if (replacement.start > lastEnd) {
                    newNodes.push({
                      type: 'text',
                      value: text.slice(lastEnd, replacement.start)
                    })
                  }
                  
                  // Add the link node
                  newNodes.push(replacement.replacement)
                  lastEnd = replacement.end
                }
                
                // Add remaining text
                if (lastEnd < text.length) {
                  newNodes.push({
                    type: 'text',
                    value: text.slice(lastEnd)
                  })
                }
                
                // Replace the text node with the new nodes
                parent.children.splice(index, 1, ...newNodes)
              }
            })
          }
        }
      ]
    },
  }
}

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
        
        // If it's already a full path, return as is
        if (cleanTarget.includes("/")) {
          return cleanTarget
        }
        
        // Look up in file index
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
        const currentDir = path.dirname(currentFile)
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
            
            visit(tree, "wikilink", (node: any) => {
              if (node.value) {
                const resolvedPath = resolveWikilink(node.value, currentFile)
                node.value = resolvedPath
              }
            })
          }
        }
      ]
    },
  }
}

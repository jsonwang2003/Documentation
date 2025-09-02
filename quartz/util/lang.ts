export function capitalize(s: string): string {
  return s.substring(0, 1).toUpperCase() + s.substring(1)
}

export function formatFolderName(name: string): string {
  // Convert "abc-xyz" format to "Abc Xyz" format
  return name
    .split('-')
    .map(word => capitalize(word.toLowerCase()))
    .join(' ')
}

export function classNames(
  displayClass?: "mobile-only" | "desktop-only",
  ...classes: string[]
): string {
  if (displayClass) {
    classes.push(displayClass)
  }
  return classes.join(" ")
}

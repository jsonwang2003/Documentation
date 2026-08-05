// getUniversityContext.js
/*
  Reusable Templater user script that infers academic context (subject & year)
  from the current file's location inside the vault.

  Rewritten to be fully self-contained — no cross-script require().
  Calls sibling scripts via tp.user.* as Templater requires.
*/

function getUniversityContext(tp, targetFile) {
  // Load siblings the Templater way
  const config     = tp.user.universityConfig(tp);
  const utils      = tp.user.universityNoteUtils(tp);

  const configLabels = config?.labels ?? {};
  const configFs     = config?.fs ?? {};

  const GENERAL_LABEL = configLabels.general;
  if (!GENERAL_LABEL) throw new Error("University config must define labels.general.");

  const UNIVERSITY_ROOT = configFs.universityRoot;
  if (!UNIVERSITY_ROOT) throw new Error("University config must define fs.universityRoot.");

  const IS_PARCIAL_ENABLED = config?.features?.parcial === true;

  const { normalizeParcial, normalizeYear } = utils;

  // No file context — return safe defaults
  if (!targetFile) {
    return { subject: GENERAL_LABEL, year: null, parcial: GENERAL_LABEL };
  }

  const parentPath = targetFile.parent?.path ?? "";
  if (!parentPath) {
    return { subject: GENERAL_LABEL, year: null, parcial: GENERAL_LABEL };
  }

  const pathParts = parentPath.split("/").filter(Boolean);
  const universityRootLower = UNIVERSITY_ROOT.toLowerCase();
  const uniIndex = pathParts.findIndex(
    (part = "") => part.toLowerCase() === universityRootLower
  );

  const relativeParts =
    uniIndex === -1 ? pathParts : pathParts.slice(uniIndex + 1);

  const frontmatterYear =
    app.metadataCache.getFileCache(targetFile)?.frontmatter?.year;
  const pathYearCandidate = relativeParts.find(
    (part = "") => normalizeYear(part, { allowLiteral: false })
  );
  const year =
    normalizeYear(frontmatterYear) ??
    normalizeYear(pathYearCandidate, { allowLiteral: false });

  const firstSegment = relativeParts[0] ?? "";
  const firstSegmentIsYear = !!normalizeYear(firstSegment, { allowLiteral: false });

  const subjectCandidate = firstSegmentIsYear
    ? relativeParts[1]
    : relativeParts[0];
  const subject = subjectCandidate || GENERAL_LABEL;

  const searchParts = firstSegmentIsYear
    ? relativeParts.slice(1)
    : relativeParts;
  const parcialCandidate = IS_PARCIAL_ENABLED
    ? searchParts.find(
        (part = "") => normalizeParcial(part) !== GENERAL_LABEL
      )
    : undefined;
  const parcial = normalizeParcial(parcialCandidate);

  return { subject, year, parcial };
}

module.exports = getUniversityContext;
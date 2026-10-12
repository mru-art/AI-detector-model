import PageHeader from "@/components/PageHeader";

export default function Privacy() {
  return (
    <div className="flex flex-col gap-10">
      <PageHeader title="Privacy" draft>
        This page will say what happens to an uploaded image.
      </PageHeader>
      <div className="max-w-2xl rounded-md bg-mist p-6 text-base leading-relaxed">
        <p>
          Placeholder. What this page says depends on how the backend handles uploads (whether
          images are stored, for how long, and whether they are ever used for training). The team
          hasn’t decided that yet, so nothing is promised here.
        </p>
      </div>
    </div>
  );
}
